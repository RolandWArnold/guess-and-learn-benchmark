import argparse
import os
import json
import stat
import subprocess
import time
from pathlib import Path

import boto3
import botocore.exceptions
import requests

# === Constants ===
REGION = "us-east-2"
ROLE_NAME = "gnl-ec2-ecr-role"
PROFILE_NAME = "gnl-ec2-ecr-profile"
POLICY_ARN = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
KEY_NAME = "gnl-keypair"
KEY_PATH = Path("~/.ssh/gnl-keypair.pem").expanduser()
GROUP_NAME = "gnl-ssh-access"
REPO_NAME = "guess-and-learn"
IMAGE_TAG = "latest"
ARCH = "linux/amd64"
DEFAULT_INSTANCE_TYPE = "g4dn.xlarge"
AMI_ID = "ami-05eb56e0befdb025f"  # Ubuntu 22.04 Deep Learning Base (EBS)

# === Utilities ===
def run(cmd, check=True):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def get_my_ip():
    return f"{requests.get('https://checkip.amazonaws.com').text.strip()}/32"

def resolve_ecr_uri():
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    return f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}"

def image_digest(image_tag):
    result = subprocess.run(f"docker inspect --format='{{{{index .RepoDigests 0}}}}' {image_tag}",
                            shell=True, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def ecr_image_digest(repo_name, tag):
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        images = ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
        return images["imageDetails"][0]["imageDigest"]
    except Exception:
        return None

# === IAM Setup ===
def setup_iam():
    iam = boto3.client("iam")

    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"[✓] Role already exists: {ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"[+] Creating role: {ROLE_NAME}")
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(assume_role_policy))

    print(f"[+] Ensuring policy attached: {POLICY_ARN}")
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN)

    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
        print(f"[✓] Instance profile exists: {PROFILE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"[+] Creating instance profile: {PROFILE_NAME}")
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)

    profile = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
    roles = [r['RoleName'] for r in profile['InstanceProfile']['Roles']]
    if ROLE_NAME not in roles:
        print(f"[+] Attaching role to instance profile")
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
    else:
        print(f"[✓] Role already attached")

    print("[+] Waiting for IAM propagation...")
    time.sleep(10)

# === Key & Network Setup ===
def setup_ec2_network():
    ec2 = boto3.client("ec2", region_name=REGION)
    ip_cidr = get_my_ip()

    if not KEY_PATH.exists():
        key_pair = ec2.create_key_pair(KeyName=KEY_NAME)
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_text(key_pair['KeyMaterial'])
        KEY_PATH.chmod(stat.S_IRUSR)
        print(f"[+] Created key at {KEY_PATH}")
    else:
        print(f"[✓] Key file exists at {KEY_PATH}")

    try:
        groups = ec2.describe_security_groups(GroupNames=[GROUP_NAME])['SecurityGroups']
        group_id = groups[0]['GroupId']
        print(f"[✓] Security group '{GROUP_NAME}' already exists")
    except botocore.exceptions.ClientError:
        print(f"[+] Creating security group '{GROUP_NAME}'")
        vpc_id = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])['Vpcs'][0]['VpcId']
        group_id = ec2.create_security_group(GroupName=GROUP_NAME, Description="Allow SSH", VpcId=vpc_id)['GroupId']
        ec2.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[{
            'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22,
            'IpRanges': [{'CidrIp': ip_cidr}]
        }])
        print(f"[✓] Opened SSH access to {ip_cidr}")

    subnet = sorted(ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])['Subnets'],
                    key=lambda s: s['AvailabilityZone'])[0]['SubnetId']
    print(f"[✓] Selected subnet {subnet}")
    return group_id, subnet

# === Docker Build and Push ===
def push_to_ecr():
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        ecr.describe_repositories(repositoryNames=[REPO_NAME])
        print(f"[✓] ECR repo exists")
    except ecr.exceptions.RepositoryNotFoundException:
        print(f"[+] Creating ECR repo")
        ecr.create_repository(repositoryName=REPO_NAME)

    repo_uri = resolve_ecr_uri()
    pw = subprocess.check_output(f"aws ecr get-login-password --region {REGION}", shell=True).decode()
    subprocess.run(["docker", "login", "--username", "AWS", "--password-stdin", repo_uri], input=pw.encode(), check=True)

    local_tag = f"{REPO_NAME}:{IMAGE_TAG}"
    remote_tag = f"{repo_uri}:{IMAGE_TAG}"

    local_digest = image_digest(local_tag)
    remote_digest = ecr_image_digest(REPO_NAME, IMAGE_TAG)

    if local_digest is None or remote_digest != local_digest:
        print("[+] Building Docker image...")
        run(f"docker build --platform {ARCH} -t {local_tag} .")
        run(f"docker tag {local_tag} {remote_tag}")
        print("[+] Digest differs — pushing image to ECR")
        run(f"docker push {remote_tag}")
    else:
        print("[✓] Image already up-to-date in ECR — skipping build and push")

# === EC2 Launch and Print Instructions ===
def get_user_data(account_id, ecr_uri):
    return f'''#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

apt-get update && apt-get install -y docker.io curl gnupg
systemctl start docker

. /etc/os-release
distribution=$ID$VERSION_ID
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{REGION}.amazonaws.com
docker pull {ecr_uri}:{IMAGE_TAG}
docker run --gpus all -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /workspace/results:/workspace/results \
  {ecr_uri}:{IMAGE_TAG} --all --workers 8 --devices cuda

shutdown -h now'''

def launch_instance(instance_type):
    ec2 = boto3.client("ec2", region_name=REGION)
    group_id, subnet_id = setup_ec2_network()
    ecr_uri = resolve_ecr_uri()
    account_id = ecr_uri.split('.')[0]
    user_data = get_user_data(account_id, ecr_uri)

    instance = ec2.run_instances(
        InstanceType=instance_type,
        ImageId=AMI_ID,
        KeyName=KEY_NAME,
        SecurityGroupIds=[group_id],
        SubnetId=subnet_id,
        IamInstanceProfile={"Name": PROFILE_NAME},
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'gnl-launch-instance'}]
        }],
        UserData=user_data,
        MinCount=1,
        MaxCount=1
    )["Instances"][0]

    instance_id = instance["InstanceId"]
    print(f"[✓] Instance launched: {instance_id}")

    print("\n--- SSH Instructions ---")
    print(f"aws ec2 describe-instances --instance-ids {instance_id} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text")
    print(f"ssh -i {KEY_PATH} ubuntu@<Public-IP>")

def terminate_all():
    ec2 = boto3.client("ec2", region_name=REGION)
    filters = [{'Name': 'tag:Name', 'Values': ['gnl-launch-instance']}]
    instances = ec2.describe_instances(Filters=filters)
    ids = [i['InstanceId'] for r in instances['Reservations'] for i in r['Instances'] if i['State']['Name'] != 'terminated']
    if ids:
        print(f"[+] Terminating instances: {' '.join(ids)}")
        ec2.terminate_instances(InstanceIds=ids)
    else:
        print("[✓] No running 'gnl-launch-instance' instances to terminate")

# === Entry ===
def main():
    parser = argparse.ArgumentParser(description="Self-managing AWS launcher for G&L")
    parser.add_argument("--prepare", action="store_true", help="Create key, security group, IAM profile")
    parser.add_argument("--push", action="store_true", help="Build and push Docker image to ECR")
    parser.add_argument("--launch", action="store_true", help="Launch EC2 instance and run")
    parser.add_argument("--terminate", action="store_true", help="Terminate all instances launched by this script")
    parser.add_argument("--print-instructions", action="store_true", help="Print full user-data and SSH instructions")
    parser.add_argument("--instance-type", type=str, default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type (default: g4dn.xlarge)")

    args = parser.parse_args()

    if args.prepare:
        setup_iam()
        setup_ec2_network()
    if args.push:
        push_to_ecr()
    if args.launch:
        launch_instance(args.instance_type)
    if args.terminate:
        terminate_all()
    if args.print_instructions:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        ecr_uri = resolve_ecr_uri()
        print("\n--- UserData Script ---\n")
        print(get_user_data(account_id, ecr_uri))
        print("\n--- SSH Instructions ---\n")
        print("Use AWS Console or describe-instances to get public IP.")
        print(f"ssh -i {KEY_PATH} ubuntu@<Public-IP>")

if __name__ == "__main__":
    main()
