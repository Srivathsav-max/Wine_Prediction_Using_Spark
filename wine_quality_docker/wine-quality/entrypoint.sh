#!/bin/bash

# Set environment variables
export SPARK_LOCAL_IP=$(hostname -i)
export SPARK_PUBLIC_DNS=$(hostname -f)

# Start supervisord
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf
