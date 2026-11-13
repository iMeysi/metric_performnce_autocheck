# OPTIONAL; requires admin approval. Replace the namespace before applying.
# cluster-monitoring-view exposes cluster-wide monitoring data; prefer a scoped
# role and endpoint if your security policy requires namespace isolation.
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: pad-omni-monitoring-view
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-monitoring-view
subjects:
  - kind: ServiceAccount
    name: pad-omni
    namespace: pad-omni
