# ✅ AWS Sandbox t3.large Deployment - READY!

## 🎯 Your Setup is Optimized!

All files have been updated for **AWS Sandbox** deployment with **t3.large** instance.

---

## 📋 What Changed

### Updated Files:
1. ✅ **QUICK_START_AWS.md** - Main deployment guide (optimized for t3.large)
2. ✅ **ec2-user-data.sh** - Auto-installation script (updated paths)
3. ✅ **AWS_SANDBOX_T3LARGE_GUIDE.md** - NEW quick reference guide
4. ✅ **cloudformation-template.yaml** - Default instance type set to t3.large
5. ✅ All cost references updated ($74/month)

### Key Changes:
- Removed Oracle Cloud references from quick start
- Updated SSH key name: `aflac-assist-key` → `rag-app-key`
- Updated application paths: `/opt/aflac-assist` → `/home/ubuntu/rag-tutorial-v2`
- Updated service name: `aflac-assist.service` → `rag-app.service`
- Added AWS sandbox-specific guidance
- Simplified for single user/small team use
- Added t3.large memory monitoring guides

---

## 🚀 Quick Start (3 Steps)

### Step 1: Read the Guide
```
Open: QUICK_START_AWS.md
```
This has complete step-by-step instructions for AWS Console.

### Step 2: Deploy (15 minutes)
1. Log into AWS Sandbox account
2. Create EC2 instance (t3.large)
3. Use the User Data script from the guide
4. Wait for auto-installation

### Step 3: Access
```
http://YOUR_PUBLIC_IP:8000
```

---

## 💻 Instance Specifications

### t3.large Details:
```
CPU: 2 vCPU (Intel Xeon)
RAM: 8 GB
Storage: 100 GB SSD (gp3)
Network: Up to 5 Gbps
Cost: ~$74/month
```

### What It Handles:
- 1-5 concurrent users
- PDFs up to 100 pages
- Total corpus: 500-1,000 pages
- Response time: 3-5 seconds
- Uptime: 24/7

### Resource Usage:
```
Ollama:           ~500 MB
Mistral model:    ~4-5 GB
nomic-embed-text: ~500 MB
ChromaDB:         ~300 MB
FastAPI:          ~200 MB
Ubuntu:           ~500 MB
---------------------------------
Total:            ~6.5 GB / 8 GB
Available:        ~1.5 GB buffer
```

---

## 🔧 User Data Script (Auto-Installation)

Copy this into **Advanced Details → User Data** when creating the instance:

```bash
#!/bin/bash
set -e

echo "=== Starting RAG Application Setup ==="
apt-get update
apt-get upgrade -y

apt-get install -y curl git python3-pip python3-venv htop

curl -fsSL https://ollama.com/install.sh | sh
sleep 10

ollama pull nomic-embed-text
ollama pull mistral

cd /home/ubuntu
git clone https://github.com/somanshushekhar/rag-tutorial-v2.git
chown -R ubuntu:ubuntu rag-tutorial-v2

cd rag-tutorial-v2
sudo -u ubuntu python3 -m pip install -r requirements.txt

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

echo "=== Installation Complete ==="
```

---

## 📊 Monitoring Your t3.large Instance

### Check Memory Usage:
```bash
# SSH into instance
ssh -i ~/.ssh/rag-app-key.pem ubuntu@YOUR_IP

# Check memory
free -h

# Monitor in real-time
watch -n 2 free -h
```

### Expected Memory Usage:
- **Idle:** ~2-3 GB
- **Processing PDF:** ~5-6 GB
- **During Query:** ~6-7 GB
- **Peak:** ~7 GB max

### Warning Signs:
- 🟡 Memory > 7 GB: Monitor closely
- 🟠 Memory > 7.5 GB: Prepare to upgrade
- 🔴 Memory > 7.8 GB or OOM errors: Upgrade NOW

---

## 🔄 Upgrade Path (If Needed)

### When to Upgrade to t3.xlarge:
- More than 5 concurrent users
- Large PDFs (> 100 pages)
- Total corpus > 1,000 pages
- Memory consistently > 90%
- OOM crashes

### How to Upgrade:
1. AWS Console → EC2 → Instances
2. Select instance → Stop
3. Actions → Instance Settings → Change Instance Type
4. Select **t3.xlarge** (4 vCPU, 16 GB RAM)
5. Start instance

**New cost:** ~$121/month

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] Instance state: Running
- [ ] Can SSH into instance
- [ ] Ollama service: Active
- [ ] RAG app service: Active
- [ ] Models downloaded (run `ollama list`)
- [ ] Can access http://YOUR_IP:8000
- [ ] Upload page loads
- [ ] Can upload a PDF
- [ ] Can chat about the document
- [ ] Memory usage < 7 GB
- [ ] No errors in logs: `sudo journalctl -u rag-app -n 50`

---

## 🐛 Troubleshooting

### Can't access http://IP:8000

```bash
# Check security group in AWS Console
# Port 8000 must be open to 0.0.0.0/0

# Check if app is running
sudo systemctl status rag-app

# Check if port is listening
sudo netstat -tuln | grep 8000
```

### Application Slow/Crashing

```bash
# Check memory
free -h

# If using swap or > 7.5 GB, upgrade to t3.xlarge
```

### Models Not Downloaded

```bash
# Check models
ollama list

# If missing, pull manually
ollama pull nomic-embed-text
ollama pull mistral

# Restart app
sudo systemctl restart rag-app
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| **QUICK_START_AWS.md** | Step-by-step deployment guide |
| **AWS_SANDBOX_T3LARGE_GUIDE.md** | Quick reference for t3.large |
| **ec2-user-data.sh** | Auto-installation script |
| **AWS_DEPLOYMENT_GUIDE.md** | Advanced deployment options |
| **DEPLOYMENT_CHECKLIST.md** | Deployment checklist |

---

## 💡 Best Practices for AWS Sandbox

### 1. Tag Your Resources:
```
Project: RAG-Application
Environment: Sandbox
Owner: your-name
```

### 2. Monitor Costs:
- Set budget alert in AWS Console
- Stop instance when not using
- Use CloudWatch for monitoring

### 3. Security:
- Change SSH security group to "My IP" only
- Don't share your SSH key
- Consider adding basic authentication

### 4. Backups:
```bash
# Backup ChromaDB weekly
tar -czf backup-$(date +%Y%m%d).tar.gz ~/rag-tutorial-v2/chroma/
```

---

## 🎯 Summary

**Your deployment is ready!**

- **Instance:** t3.large (2 vCPU, 8 GB RAM)
- **Cost:** ~$74/month (sandbox may cover)
- **Capacity:** 1-5 users, 500-1,000 pages
- **Deployment:** 15 minutes with auto-script
- **Upgrade:** t3.xlarge if needed

**Next Step:** Open `QUICK_START_AWS.md` and start deploying! 🚀

---

## 🆘 Need Help?

### Common Commands:

```bash
# SSH into instance
ssh -i ~/.ssh/rag-app-key.pem ubuntu@YOUR_IP

# Check status
sudo systemctl status rag-app

# View logs
sudo journalctl -u rag-app -f

# Restart application
sudo systemctl restart rag-app

# Check memory
free -h

# Check models
ollama list
```

---

**Everything is configured for t3.large on AWS Sandbox! Follow QUICK_START_AWS.md to deploy!** ✅
