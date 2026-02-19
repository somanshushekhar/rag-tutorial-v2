# AWS Deployment Guide for Aflac Assist Chat

## 📋 Prerequisites

### 1. AWS Account Requirements
- Active AWS account
- AWS CLI installed and configured
- IAM user with permissions:
  - EC2 (full access)
  - CloudFormation (optional)
  - IAM (create roles)

### 2. Local Requirements
- Git
- SSH key pair for EC2 access

---

## 🚀 Deployment Methods

### **Method 1: Automated Deployment (Recommended)**

#### Step 1: Create EC2 Key Pair
```bash
# In AWS Console: EC2 → Key Pairs → Create Key Pair
# Download the .pem file
chmod 400 your-key.pem
```

#### Step 2: Deploy Using CloudFormation
```bash
aws cloudformation create-stack \
  --stack-name aflac-assist-chat \
  --template-body file://cloudformation-template.yaml \
  --parameters ParameterKey=KeyName,ParameterValue=your-key-name \
               ParameterKey=InstanceType,ParameterValue=t3.large
```

#### Step 3: Get Instance IP
```bash
aws cloudformation describe-stacks \
  --stack-name aflac-assist-chat \
  --query 'Stacks[0].Outputs'
```

#### Step 4: Access Application
```
http://<PUBLIC_IP>:8000
```

---

### **Method 2: Manual EC2 Setup**

#### Step 1: Launch EC2 Instance

**AWS Console:**
1. Go to EC2 Dashboard → Launch Instance
2. **Name:** Aflac-Assist-Chat
3. **AMI:** Ubuntu Server 22.04 LTS
4. **Instance Type:** 
   - **Minimum:** t3.large (2 vCPU, 8 GB RAM) - $0.0832/hour
   - **Recommended:** t3.xlarge (4 vCPU, 16 GB RAM) - $0.1664/hour
   - **GPU Option:** g4dn.xlarge (4 vCPU, 16 GB RAM, GPU) - $0.526/hour
5. **Key Pair:** Select or create
6. **Storage:** 100 GB gp3
7. **Security Group:** Create new with rules:
   - SSH (22) - Your IP only
   - HTTP (80) - 0.0.0.0/0
   - HTTPS (443) - 0.0.0.0/0
   - Custom TCP (8000) - 0.0.0.0/0

#### Step 2: Connect to Instance
```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
```

#### Step 3: Run Installation Script
```bash
# Upload the ec2-user-data.sh script
scp -i your-key.pem ec2-user-data.sh ubuntu@<PUBLIC_IP>:~/

# SSH into instance
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

# Run setup
sudo bash ec2-user-data.sh
```

#### Step 4: Verify Installation
```bash
# Check Ollama
sudo systemctl status ollama

# Check application
sudo systemctl status aflac-assist

# View logs
sudo journalctl -u aflac-assist -f
```

#### Step 5: Access Application
```
http://<PUBLIC_IP>:8000
```

---

### **Method 3: Docker Deployment**

#### Step 1: Launch EC2 with Docker
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
```

#### Step 2: Clone and Build
```bash
cd /opt
sudo git clone https://github.com/somanshushekhar/rag-tutorial-v2.git
cd rag-tutorial-v2

# Build and run
sudo docker-compose up -d

# View logs
sudo docker-compose logs -f
```

---

## 💰 Cost Estimation

### EC2 Instance Costs (us-east-1)

| Instance Type | vCPU | RAM | GPU | Cost/Hour | Cost/Month* |
|---------------|------|-----|-----|-----------|-------------|
| t3.large | 2 | 8 GB | No | $0.0832 | $60 |
| t3.xlarge | 4 | 16 GB | No | $0.1664 | $120 |
| t3.2xlarge | 8 | 32 GB | No | $0.3328 | $240 |
| g4dn.xlarge | 4 | 16 GB | Yes | $0.526 | $380 |

*Based on 24/7 uptime (730 hours/month)

### Additional Costs
- **EBS Storage:** $0.08/GB/month (100 GB = $8/month)
- **Data Transfer:** $0.09/GB (first 10 TB out)
- **Elastic IP:** Free if attached

**Total Estimated Monthly Cost:** $130-$390

---

## 🔒 Security Hardening

### 1. Restrict SSH Access
```bash
# In Security Group, change SSH source from 0.0.0.0/0 to your IP
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 22 \
  --cidr <YOUR_IP>/32
```

### 2. Enable HTTPS

#### Install Nginx + Certbot
```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/aflac-assist
```

**Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/aflac-assist /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 3. Add Authentication

Update `app.py`:
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from secrets import compare_digest

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials):
    correct_username = compare_digest(credentials.username, "admin")
    correct_password = compare_digest(credentials.password, "your-secure-password")
    return correct_username and correct_password

@app.post("/upload")
async def upload(credentials: HTTPBasicCredentials = Depends(security)):
    if not verify_credentials(credentials):
        raise HTTPException(401, "Invalid credentials")
    # ... rest of code
```

---

## 🔧 Post-Deployment Configuration

### 1. Upload PDF Files
```bash
# From your local machine
scp -i your-key.pem data/*.pdf ubuntu@<PUBLIC_IP>:/opt/aflac-assist/data/
```

### 2. Ingest Documents
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

# Navigate to app directory
cd /opt/aflac-assist
source .venv/bin/activate

# Ingest PDFs
python populate_database.py
```

### 3. Test the Application
```bash
# Test embedding
python test_setup.py

# Test RAG
python test_rag.py
```

---

## 📊 Monitoring & Maintenance

### 1. View Logs
```bash
# Application logs
sudo journalctl -u aflac-assist -f

# Ollama logs
sudo journalctl -u ollama -f

# Docker logs (if using Docker)
sudo docker-compose logs -f
```

### 2. Restart Services
```bash
sudo systemctl restart ollama
sudo systemctl restart aflac-assist
```

### 3. Update Application
```bash
cd /opt/aflac-assist
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart aflac-assist
```

### 4. Backup ChromaDB
```bash
# Create backup
tar -czf chroma-backup-$(date +%Y%m%d).tar.gz /opt/aflac-assist/chroma/

# Upload to S3
aws s3 cp chroma-backup-*.tar.gz s3://your-backup-bucket/
```

---

## 🚨 Troubleshooting

### Issue: Ollama Not Starting
```bash
# Check status
sudo systemctl status ollama

# View logs
sudo journalctl -u ollama -n 50

# Restart
sudo systemctl restart ollama
```

### Issue: Out of Memory
```bash
# Check memory usage
free -h

# Solution: Upgrade to larger instance (t3.2xlarge)
```

### Issue: Slow Embeddings
```bash
# Use GPU instance (g4dn.xlarge)
# Or use smaller model
export OLLAMA_EMBED_MODEL=all-minilm
```

### Issue: Application Not Accessible
```bash
# Check security group allows port 8000
# Check application is running
sudo systemctl status aflac-assist

# Check firewall
sudo ufw status
```

---

## 🎯 Production Checklist

- [ ] EC2 instance launched with correct size
- [ ] Security groups configured
- [ ] SSH key pair saved securely
- [ ] Ollama installed and running
- [ ] Models pulled (nomic-embed-text, mistral)
- [ ] Application deployed and running
- [ ] PDF files uploaded
- [ ] Documents ingested (ChromaDB populated)
- [ ] HTTPS configured (optional)
- [ ] Authentication added (optional)
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Domain name configured (optional)

---

## 📞 Support

For issues, check:
1. Application logs: `sudo journalctl -u aflac-assist -f`
2. Ollama logs: `sudo journalctl -u ollama -f`
3. GitHub Issues: https://github.com/somanshushekhar/rag-tutorial-v2/issues
