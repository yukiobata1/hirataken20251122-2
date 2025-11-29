#!/bin/bash
# Check status of EPSR simulation
# Usage: ./epsr_status.sh

PIDFILE="logs/epsr.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found. EPSR is not running (or was not started with run_epsr_nohup.sh)"
    exit 1
fi

PID=$(cat "$PIDFILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "EPSR is running (PID: $PID)"

    # Find the latest log file
    LATEST_LOG=$(ls -t logs/epsr_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Log file: $LATEST_LOG"
        echo ""
        echo "=== Last 20 lines of log ==="
        tail -20 "$LATEST_LOG"
        echo ""
        echo "=== To follow log in real-time ==="
        echo "tail -f $LATEST_LOG"
    fi
else
    echo "EPSR is not running (PID $PID is dead)"
    echo "You may want to check the log files in logs/"
    rm -f "$PIDFILE"
fi
