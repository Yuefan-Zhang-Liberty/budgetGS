#!/bin/bash
echo "=== per-node GPU free (name / total-used / state) ==="
scontrol show nodes | awk '
/NodeName=/{
  name=$1; sub("NodeName=","",name); state=""; gres=""; gused=""
}
/State=/{ for(i=1;i<=NF;i++){ if($i ~ /^State=/){ state=$i; sub("State=","",state) } } }
/Gres=/{ for(i=1;i<=NF;i++){ if($i ~ /^gres:/){gres=$i} } }
/GresUsed=/{ for(i=1;i<=NF;i++){ if($i ~ /^GresUsed=/){gused=$i} } }
/^$/{ if(name!=""){ print name"  state="state"  gres="gres"  used="gused; name="" } }
END{ if(name!=""){ print name"  state="state"  gres="gres"  used="gused } }
'
