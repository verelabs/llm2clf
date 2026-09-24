#!/usr/bin/env bash
# Launches one p5.4xlarge (1x H100 80GB) that terminates itself after MAX_MINUTES, with SSH open only to this machine's IP.
set -euo pipefail

REGION=${REGION:-us-west-2}
TYPE=${TYPE:-p5.4xlarge}
MAX_MINUTES=${MAX_MINUTES:-240}
NAME=jevlocal-bench
KEY=jevlocal
KEY_FILE=~/.ssh/$KEY.pem
AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY" >/dev/null 2>&1; then
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY" --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=$NAME Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name $NAME --description "jevlocal benchmark SSH" --vpc-id "$VPC" --query GroupId --output text)
fi
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null 2>&1 || true

USER_DATA=$(printf '#!/bin/bash\nshutdown -h +%s\n' "$MAX_MINUTES")
for SUBNET in $(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC" Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text); do
  if ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
      --security-group-ids "$SG" --subnet-id "$SUBNET" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=400,VolumeType=gp3,Throughput=1000,Iops=6000,DeleteOnTermination=true}' \
      --user-data "$USER_DATA" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
      --query 'Instances[0].InstanceId' --output text 2>/tmp/jevlocal-launch.err); then
    break
  fi
  echo "no capacity in $SUBNET: $(tail -1 /tmp/jevlocal-launch.err)" >&2
  ID=""
done
[ -n "$ID" ] || { echo "launch failed in every zone" >&2; exit 1; }

aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$ID $IP"
