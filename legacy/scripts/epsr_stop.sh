#!/bin/bash
# Stop EPSR simulation
# Usage: ./epsr_stop.sh

PIDFILE="logs/epsr.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found. EPSR may not be running."
    exit 1
fi

PID=$(cat "$PIDFILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping EPSR (PID: $PID)..."
    kill $PID
    sleep 2

    # Check if it's still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process still running, forcing kill..."
        kill -9 $PID
    fi

    echo "EPSR stopped."
    rm -f "$PIDFILE"
else
    echo "EPSR is not running (PID $PID is dead)"
    rm -f "$PIDFILE"
fi
