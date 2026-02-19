# ⚡ Quick Start: Deploy to AWS in 15 Minutes

## 🎯 What You'll Need

1. **AWS Sandbox Account** (provided by your organization)
2. **SSH Key** (we'll create this)
3. **15-20 minutes of your time**

> 💡 **Instance Type:** t3.large (2 vCPU, 8 GB RAM) - Cost: ~$74/month or covered by your sandbox account

---

## 📝 Step-by-Step Instructions

### **Step 1: Create EC2 Instance** (5 minutes)

#### 1.1 Log into AWS Console
- Go to https://console.aws.amazon.com
- Navigate to **EC2 Dashboard**

#### 1.2 Create SSH Key Pair
1. Click **Key Pairs** (left menu)
2. Click **Create key pair**
   - Name: `aflac-assist-key`
   - Type: **RSA**
   - Format: **pem**
3. Click **Create** → Downloads `aflac-assist-key.pem`
4. **Save this file securely!**

#### 1.3 Launch Instance
1. Click **Launch Instance** (orange button)
2. **Name:** `RAG-Application`
3. **Application and OS Images:**
   - Quick Start: **Ubuntu**
   - AMI: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
   - Architecture: **64-bit (x86)**
4. **Instance type:**
   - Click dropdown and select **t3.large**
   - Specs: 2 vCPU, 8 GB RAM
   - ⚠️ Cost: $0.0832/hour (~$61/month if not covered by sandbox)
5. **Key pair:**
   - Select **rag-app-key** (created above)
6. **Network settings:**
   - Click **Edit**
   - **Firewall (security groups):** Create security group
   - **Security group name:** `aflac-assist-sg`
   - Add rules:
     - ✅ SSH (22) - **My IP** (automatically fills your IP)
     - ✅ HTTP (80) - **Anywhere (0.0.0.0/0)**
     - ✅ HTTPS (443) - **Anywhere (0.0.0.0/0)**
     - ➕ Click **Add security group rule**
       - Type: **Custom TCP**
       - Port: **8000**
       - Source: **Anywhere (0.0.0.0/0)**
7. **Configure storage:**
   - Change **Size (GiB):** to **100**
   - Volume type: **gp3**
8. **Advanced details (optional):**
   - If you want automatic setup, paste content from `ec2-user-data.sh` into **User data** field
9. Click **Launch instance**

#### 1.4 Get Public IP
1. Click **View all instances**
2. Wait for **Instance state** to show **Running** (~2 minutes)
3. Copy the **Public IPv4 address** (e.g., 54.123.45.67)

---

### **Step 2: Connect to Instance** (2 minutes)

#### For Windows:
```powershell
# Move to directory with key file
cd Downloads

# Set permissions (if using Git Bash or WSL)
chmod 400 aflac-assist-key.pem

# Connect
ssh -i aflac-assist-key.pem ubuntu@54.123.45.67
```

#### For Mac/Linux:
```bash
# Move key to .ssh folder
mv ~/Downloads/aflac-assist-key.pem ~/.ssh/
chmod 400 ~/.ssh/aflac-assist-key.pem

# Connect
ssh -i ~/.ssh/aflac-assist-key.pem ubuntu@54.123.45.67
```

**When prompted "Are you sure you want to continue connecting?"** → Type `yes`

---

### **Step 3: Install Application** (8 minutes)

#### 3.1 Run Installation Commands

Paste this entire block into your SSH session:

```bash
# Update system
sudo apt-get update

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
sudo systemctl start ollama
sudo systemctl enable ollama

# Wait for Ollama to start
sleep 5

# Pull AI models (this takes ~5 minutes)
ollama pull nomic-embed-text
ollama pull mistral

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv git

# Clone your repository
cd /opt
sudo git clone https://github.com/somanshushekhar/rag-tutorial-v2.git aflac-assist
cd aflac-assist

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Create directories
mkdir -p data chroma

# Change ownership
sudo chown -R ubuntu:ubuntu /opt/aflac-assist
```

#### 3.2 Create Systemd Service

```bash
sudo tee /etc/systemd/system/aflac-assist.service > /dev/null <<EOF
[Unit]
Description=Aflac Assist Chat
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/aflac-assist
Environment="PATH=/opt/aflac-assist/.venv/bin"
Environment="OLLAMA_BASE_URL=http://127.0.0.1:11434"
ExecStart=/opt/aflac-assist/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

#### 3.3 Start Application

```bash
sudo systemctl daemon-reload
sudo systemctl start aflac-assist
sudo systemctl enable aflac-assist
```

#### 3.4 Verify It's Running

```bash
sudo systemctl status aflac-assist
```

You should see **"Active: active (running)"** in green.

---

### **Step 4: Access Your Application** (1 minute)

Open your browser and go to:

```
http://54.123.45.67:8000
```

(Replace with your actual EC2 public IP)

**You should see the upload page!** 🎉

---

### **Step 5: Upload Your First PDF**

1. Click **Choose Files**
2. Select a PDF document
3. Click **Upload & Ingest**
4. Wait ~30 seconds for processing
5. Click **Go to Chat Interface**
6. Ask a question about your document!

---

## 🔧 Troubleshooting

### ❌ Can't connect to EC2
**Problem:** SSH connection refused  
**Solution:** Check security group allows SSH from your IP

```bash
# Get your current IP
curl https://checkip.amazonaws.com

# Update security group in AWS Console to allow this IP
```

---

### ❌ Page not loading (http://IP:8000)
**Problem:** Application not running  
**Solution:** Check status and logs

```bash
# Check if running
sudo systemctl status aflac-assist

# View logs
sudo journalctl -u aflac-assist -n 50

# Restart service
sudo systemctl restart aflac-assist
```

---

### ❌ Ollama not working
**Problem:** "Connection refused" to Ollama  
**Solution:** Check Ollama service

```bash
# Check Ollama status
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Check if models are downloaded
ollama list
```

---

### ❌ Out of memory
**Problem:** Application crashes  
**Solution:** Upgrade to larger instance

1. Stop instance in AWS Console
2. Right-click → **Instance settings** → **Change instance type**
3. Select **t3.xlarge** (16 GB RAM) or **t3.2xlarge** (32 GB RAM)
4. Start instance

---

## 💡 Next Steps

### 1. **Enable HTTPS** (Optional but Recommended)

```bash
# Install Nginx and Certbot
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Get free SSL certificate (requires domain name)
sudo certbot --nginx -d yourdomain.com
```

### 2. **Add Authentication** (Security)

Edit `app.py` and add:
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

@app.post("/upload")
async def upload(credentials: HTTPBasicCredentials = Depends(security)):
    # Add auth check here
```

### 3. **Set Up Backups**

```bash
# Create backup script
cat > /opt/backup-chromadb.sh <<'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf /tmp/chroma-backup-$DATE.tar.gz /opt/aflac-assist/chroma/
# Upload to S3 (optional)
# aws s3 cp /tmp/chroma-backup-$DATE.tar.gz s3://your-bucket/
EOF

chmod +x /opt/backup-chromadb.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/backup-chromadb.sh") | crontab -
```

### 4. **Monitor Costs**

- Go to **AWS Billing Dashboard**
- Set up **Budget Alert** for $150/month
- Enable **Cost Explorer**

---

## 📊 Cost Summary

| Resource | Cost/Month |
|----------|------------|
| EC2 t3.large | $60.74 |
| EBS 100 GB | $8.00 |
| Data Transfer | ~$5.00 |
| **TOTAL** | **~$74/month** |

**Ways to reduce costs:**
- ⏸️ Stop instance when not in use: `$0/month`
- 📅 Reserved Instance (1 year): Save 30% → `$52/month`
- 🎯 Spot Instance: Save 70% → `$22/month` (can be interrupted)

---

## 🆘 Need Help?

### Check Logs
```bash
# Application logs
sudo journalctl -u aflac-assist -f

# Ollama logs  
sudo journalctl -u ollama -f
```

### Restart Everything
```bash
sudo systemctl restart ollama
sudo systemctl restart aflac-assist
```

### SSH Back In
```bash
ssh -i ~/.ssh/aflac-assist-key.pem ubuntu@YOUR_EC2_IP
```

---

## ✅ Success Checklist

- [ ] EC2 instance created (t3.large)
- [ ] SSH key saved securely
- [ ] Connected via SSH
- [ ] Ollama installed and running
- [ ] Models downloaded (nomic-embed-text, mistral)
- [ ] Application running on port 8000
- [ ] Can access http://YOUR_IP:8000
- [ ] PDF uploaded successfully
- [ ] Chat interface works

---

## 🎉 Congratulations!

Your Aflac Assist Chat is now running on AWS!

**Application URL:** `http://YOUR_EC2_IP:8000`

**Estimated Cost:** $74/month (or less if you stop when not using)
