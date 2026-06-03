#!/usr/bin/env python3
"""
simulate.py - Simulação de montoramento de voo em tempo real.

Envia um voo completo para a API e reproduz os alertas no terminal
na velocidade configurada, como se os dados chegassem em tempo real.

Uso:
    poetry run python simulate.py <caminho_do_csv>
    poetry run python simulate.py <caminho_do_csv> --speed 20
    poetry run python simulate.py <caminho_do_csv> --url http://localhost:8000
    
"""

import argparse
import sys
import time

import httpx
import pandas as pd
from pathlib import Path

#Cores ANSI para o terminal
RED = "\033[91m"
GREEN =  "\033[92m"
YELLOW =  "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulação de detecção de falhas de motor em tempo real"
    )
    parser.add_argument("csv_path", help="Caminho para o CSV de voo")
    parser.add_argument(
    "--url", default="http://localhost:8000",
    help="URL base da API (padrão: http://localhost:8000)"
    )
    parser.add_argument(
        "--speed", type=float, default=10.0,
        help="Multiplicador de velocidade da simulação (padrão: 10x)"
    )
    return parser.parse_args()

def df_to_readings(df: pd.DataFrame) -> list[dict]:
    required = [
        "timestamp", "hud_throttle", "err_vel_z", "alt_global",
        "alt_gps_fix", "vel_z_meas", "vel_z_local", "pos_z_local", "vel_z_twist",
    ]
    cols = required + (["aspd_meas"] if "aspd_meas" in df.columns else [])
    return df[cols].to_dict(orient="records")

def check_api(url: str) -> bool:
    try:
        r=httpx.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False
    
def send_flight(url: str, flight_id: str, readings: list[dict]) -> dict:
    r = httpx.post(
        f"{url}/predict",
        json={"flight_id": flight_id, "readings": readings},
        timeout=300 #FFT de voos longos pode demorar
    )
    r.raise_for_status()
    return r.json()

def replay(events: list[dict], speed: float, fault_start: float | None):
    """
    Reproduz os eventos no terminal na velocidade configurada.
    """
    print(f"\n {'TEMPO':>8} STATUS")
    print(f" {'-'*30}")

    t_sim_start = time.time()
    t_flight_start = events[0]["timestamp"]
    in_anomaly = False

    for event in events:
        #Aguarda até o momento correto na simulação
        flight_elapsed = event["timestamp"] - t_flight_start
        sim_elapsed = time.time() - t_sim_start
        wait = (flight_elapsed/speed) - sim_elapsed
        if wait > 0:
            time.sleep(wait)

        t = event["timestamp"]

        #Só imprime quando o estado muda - evita flood no terminal
        if event["is_anomaly"] and not in_anomaly:
            in_anomaly = True
            marker = ""
            if fault_start and t >= fault_start:
                marker = f"← falha real em t={fault_start:.1f}s"
            print(f"  {t:>8.2f}s  {RED}{BOLD}⚠  ANOMALIA DETECTADA{RESET}{marker}")
            
        elif not event["is_anomaly"] and in_anomaly:
            in_anomaly = False
            print(f"  {t:>8.2f}s  {GREEN}✓  retornou ao normal{RESET}")

    if in_anomaly:
        last_t = events[-1]['timestamp']
        print(f"  {last_t:>8.2f}s  {RED}⚠  fim do voo em anomalia{RESET}")

def print_summary(body: dict, df: pd.DataFrame):
    total= len(body["events"])
    anomalies = body["anomalies_detected"]
    rate = anomalies / total if total > 0 else 0

    print(f"\n  {'─' * 40}")
    print(f"  {BOLD}RESUMO{RESET}")
    print(f"  Eventos analisados:   {total}")
    print(f"  Anomalias detectadas: {anomalies}  ({rate:.1%})")

    if body["first_anomaly_at"]:
        print(f"  Primeiro alerta:      t={body['first_anomaly_at']:.2f}s")

        if "target_fault" in df.columns:
            fault_rows = df[df["target_fault"] == 1]
            if not fault_rows.empty:
                fault_start = fault_rows["timestamp"].min()
                latency = body["first_anomaly_at"] - fault_start
                print(f"  Início real da falha: t={fault_start:.2f}s")
                color = GREEN if latency < 5 else YELLOW if latency < 30 else RED
                print(f"  Latência de detecção: {color}{latency:+.3f}s{RESET}")


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"{RED}Arquivo não encontrado: {csv_path}{RESET}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    duration = df["timestamp"].max() - df["timestamp"].min()
    has_fault = "target_fault" in df.columns and df["target_fault"].any()
    fault_start = df[df["target_fault"] == 1]["timestamp"].min() if has_fault else None

    print(f"\n{BOLD}  Motor Failure Detection — Simulação{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Voo:        {csv_path.name}")
    print(f"  Duração:    {duration:.1f}s  ({len(df):,} amostras)")
    print(f"  Falha real: {'t=' + f'{fault_start:.1f}s' if fault_start else 'nenhuma'}")
    print(f"  Velocidade: {args.speed}x  (simulação de {duration/args.speed:.0f}s)")
    print(f"  {'─' * 40}")

    print(f"\n  Verificando API em {args.url}...", end=" ", flush=True)
    if not check_api(args.url):
        print(f"{RED}offline{RESET}")
        print("  Suba o servidor com: sudo docker compose up")
        sys.exit(1)
    print(f"{GREEN}online{RESET}")

    print(f"  Enviando {len(df):,} amostras para análise...", end=" ", flush=True)
    try:
        body = send_flight(args.url, csv_path.stem, df_to_readings(df))
    except httpx.HTTPStatusError as e:
        print(f"{RED}erro {e.response.status_code}{RESET}")
        print(f"  Detalhe: {e.response.text}")
        sys.exit(1)
    print(f"{GREEN}{len(body['events'])} predições recebidas{RESET}")

    print(f"\n  {BOLD}▶ Reproduzindo em {args.speed}x...{RESET}")
    replay(body["events"], args.speed, fault_start)
    print_summary(body, df)
    print()

if __name__ == "__main__":
    main()