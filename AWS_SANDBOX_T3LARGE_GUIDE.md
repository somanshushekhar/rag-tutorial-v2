# 🚀 AWS Sandbox t3.large Deployment - Quick Reference

## ✅ Optimized for Your Setup

- **Platform:** AWS Sandbox Account
- **Instance Type:** t3.large (2 vCPU, 8 GB RAM)
- **Cost:** ~$74/month (or covered by sandbox account)
- **Deployment Time:** 15-20 minutes

---

## 📋 Pre-Deployment Checklist

- [ ] AWS Sandbox account access
- [ ] Can access AWS Console (console.aws.amazon.com)
- [ ] Browser ready
- [ ] 15-20 minutes available
- [ ] Note-taking app ready (for IP address, etc.)

---

## ⚡ Quick Start (Follow QUICK_START_AWS.md)

### Phase 1: AWS Console Setup (7 minutes)
1. ✅ Log into AWS Console
2. ✅ Create SSH key pair (`rag-app-key.pem`)
3. ✅ Launch t3.large instance
4. ✅ Configure security group (ports 22, 80, 443, 8000)
5. ✅ Add User Data script for auto-installation
6. ✅ Get public IP address

### Phase 2: Connect & Verify (3 minutes)
1. ✅ SSH into instance
2. ✅ Wait for auto-installation to complete
3. ✅ Verify Ollama is running
4. ✅ Verify app is running

### Phase 3: Test Application (5 minutes)
1. ✅ Access http://YOUR_IP:8000
2. ✅ Upload a PDF
3. ✅ Test chat interface
4. ✅ Verify responses

---

## 🔧 t3.large Configuration Details

### What You Get:
- **vCPU:** 2 cores
- **RAM:** 8 GB
- **Storage:** 100 GB SSD (gp3)
- **Network:** Up to 5 Gbps
- **Monthly Cost:** ~$74 (may be covered by sandbox)

### What It Can Handle:
- ✅ 1-5 concurrent users
- ✅ PDFs up to 100 pages
- ✅ Total corpus: 500-1,000 pages
- ✅ 3-5 second response time
- ✅ 2-3 models loaded

### Resource Usage:
- **Ollama:** ~500 MB
- **Mistral model:** ~4-5 GB (when loaded)
- **nomic-embed-text:** ~500 MB
- **ChromaDB:** ~200-500 MB
- **FastAPI:** ~200 MB
- **Ubuntu OS:** ~500 MB
- **Total:** ~6-7 GB (leaves 1-2 GB buffer)

---

## 📊 Memory Management

### Monitor Memory:
```bash
# Check current usage
free -h

# Watch in real-time
watch -n 2 free -h

# Check which processes use memory
ps aux --sort=-%mem | head -10
```

### Good Signs (t3.large is sufficient):
- ✅ Memory usage < 6.5 GB
- ✅ No swap usage
- ✅ Response time < 5 seconds
- ✅ No crashes

### Warning Signs (consider upgrading):
- ⚠️ Memory usage > 7 GB
- ⚠️ Swap being used
- ⚠️ Response time > 8 seconds
- ⚠️ Occasional slowdowns

### Critical Signs (MUST upgrade):
- 🔴 Memory usage > 7.5 GB
- 🔴 Heavy swap usage
- 🔴 Application crashes
- 🔴 OOM (Out of Memory) errors
- 🔴 Response time > 15 seconds

---

## 🚀 Deployment Script (User Data)

This goes in **Advanced Details → User Data** when launching the instance:

```bash
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y curl git python3-pip python3-venv

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Wait for Ollama to start
sleep 10

# Pull required models (this takes 5-10 minutes)
ollama pull nomic-embed-text
ollama pull mistral

# Clone repository
cd /home/ubuntu
git clone https://github.com/somanshushekhar/rag-tutorial-v2.git
chown -R ubuntu:ubuntu rag-tutorial-v2

# Install Python dependencies
cd rag-tutorial-v2
sudo -u ubuntu python3 -m pip install -r requirements.txt

# Create systemd service
cat > /etc/systemd/system/rag-app.service <<EOF
[Unit]
Description=RAG Application
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rag-tutorial-v2
Environment="OLLAMA_BASE_URL=http://127.0.0.1:11434"
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rag-app
systemctl start rag-app
```

---

## 🔍 Troubleshooting

### Can't access http://IP:8000

**Check 1: Security Group**
```bash
# AWS Console → EC2 → Security Groups
# Verify port 8000 is open to 0.0.0.0/0
```

**Check 2: Application Status**
```bash
sudo systemctl status rag-app
# Should show "active (running)" in green
```

**Check 3: Firewall**
```bash
# Check if port is listening
sudo netstat -tuln | grep 8000
# Should show: tcp 0 0 0.0.0.0:8000 0.0.0.0:* LISTEN
```

**Check 4: Models Downloaded**
```bash
ollama list
# Should show nomic-embed-text and mistral
```

### Application Crashes / Out of Memory

**Immediate Fix:**
```bash
# Restart application
sudo systemctl restart rag-app

# Check memory
free -h
```

**Long-term Solution:**
```bash
# Upgrade to t3.xlarge (16 GB RAM)
# 1. Stop instance in AWS Console
# 2. Change instance type to t3.xlarge
# 3. Start instance
# New cost: ~$121/month
```

### Slow Performance

**Check CPU:**
```bash
top
# Press '1' to show all CPUs
# If both CPUs at 100%, consider upgrading
```

**Check Memory:**
```bash
free -h
# If swap is being used, you need more RAM
```

**Check Disk:**
```bash
df -h
# Should have plenty of free space
```

---

## 📈 When to Upgrade to t3.xlarge

### Upgrade if:
- More than 5 concurrent users
- Processing PDFs > 100 pages regularly
- Total corpus > 1,000 pages
- Memory usage consistently > 90%
- Need faster response times (< 2 seconds)
- Application crashes with OOM errors

### How to Upgrade:
1. AWS Console → EC2 → Instances
2. Select instance → Stop
3. Actions → Instance Settings → Change Instance Type
4. Select t3.xlarge
5. Start instance
6. Cost increases to ~$121/month

---

## 💡 Cost Optimization for Sandbox

### If Sandbox Has Budget Limits:

**Option 1: Stop When Not Using**
```bash
# Stop instance at night/weekends
# Only pay for storage (~$8/month)
# Start again when needed
```

**Option 2: Use Smaller Models**
```python
# In your code, use smaller models:
ollama pull mistral:7b-instruct-q4_0  # Smaller quantized version
```

**Option 3: Limit Concurrent Users**
```python
# Add rate limiting in app.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

---

## ✅ Success Criteria

### Your deployment is successful when:
- [ ] Can access http://YOUR_IP:8000
- [ ] Upload page loads correctly
- [ ] Can upload a PDF (< 50 pages for testing)
- [ ] PDF processes in < 2 minutes
- [ ] Can ask questions in chat
- [ ] Get relevant answers in < 5 seconds
- [ ] Memory usage < 6.5 GB
- [ ] No errors in logs

---

## 🔐 Security Reminders

### Lock Down SSH:
```bash
# In AWS Console → Security Groups
# Change SSH source from 0.0.0.0/0 to "My IP"
```

### Add HTTPS (Optional):
```bash
# Requires domain name
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Add Basic Auth (Quick Security):
```python
# Add to app.py
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
import secrets

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "your-password")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials"
        )
    return credentials.username
```

---

## 📚 Documentation References

- **Full Guide:** `QUICK_START_AWS.md`
- **Architecture:** `AWS_ARCHITECTURE_DIAGRAMS.md`
- **Deployment Checklist:** `DEPLOYMENT_CHECKLIST.md`
- **Troubleshooting:** `AWS_DEPLOYMENT_GUIDE.md`

---

## 🎯 Summary

**Instance:** t3.large (2 vCPU, 8 GB RAM)  
**Storage:** 100 GB SSD  
**Cost:** ~$74/month  
**Setup Time:** 15-20 minutes  
**Capacity:** 1-5 users, 500-1,000 pages  
**Upgrade Path:** t3.xlarge if needed ($121/month)

**Your deployment is optimized for AWS sandbox with t3.large!** ✅
