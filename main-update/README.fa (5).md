# OPTIONAL: requires Prometheus Operator and enabled user-workload monitoring.
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: pad-omni
spec:
  selector:
    matchLabels:
      app: pad-omni
  endpoints:
    - port: metrics
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: pad-omni
spec:
  groups:
    - name: pad-omni
      rules:
        - alert: MultivariateAnomaly
          expr: pad_omni_anomaly == 1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: 'Multivariate anomaly in {{ $labels.group }}'
        - alert: OmniModelUnavailable
          expr: pad_omni_ready == 0
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: 'No fresh inference for {{ $labels.group }}'
        - alert: OmniExporterDown
          expr: up{service="pad-omni"} == 0
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: 'Omni exporter scrape failed (verify service label at your site)'
