#!/bin/bash
# Wrapper script to run EPSR with nohup
# Usage: ./run_epsr_nohup.sh

LOGDIR="logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/epsr_${TIMESTAMP}.log"
PIDFILE="$LOGDIR/epsr.pid"

echo "Starting EPSR simulation in background..."
echo "Log file: $LOGFILE"
echo "PID file: $PIDFILE"

# Run with nohup and redirect output
nohup python scripts/main_epsr.py > "$LOGFILE" 2>&1 &

# Save PID
echo $! > "$PIDFILE"
echo "Process started with PID: $(cat $PIDFILE)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To stop the process:"
echo "  kill \$(cat $PIDFILE)"
echo ""
