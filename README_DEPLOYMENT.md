# 🚀 AWS Deployment Package

Complete deployment solution for **Aflac Assist Chat** on AWS.

---

## 🎯 What This Package Includes

### **✅ Bug Fix**
- Fixed Ollama port from `11435` → `11434` in `get_embedding_function.py`

### **📦 Deployment Files** (9 new files)
1. `Dockerfile` - Container definition
2. `docker-compose.yml` - Multi-container setup
3. `start.sh` - Startup script
4. `ec2-user-data.sh` - Automated EC2 setup
5. `cloudformation-template.yaml` - Infrastructure as Code
6. `deploy-to-aws.sh` - One-command deployment

### **📚 Documentation** (4 comprehensive guides)
7. `QUICK_START_AWS.md` - **START HERE** (15-minute guide)
8. `AWS_DEPLOYMENT_GUIDE.md` - Complete reference
9. `AWS_ARCHITECTURE_DIAGRAMS.md` - Visual diagrams
10. `DEPLOYMENT_CHECKLIST.md` - Step-by-step checklist
11. `DEPLOYMENT_SUMMARY.md` - Overview
12. `README_DEPLOYMENT.md` - This file

---

## ⚡ Quick Start (3 Steps)

### **Step 1: Choose Your Path**

**New to AWS?** → Read `QUICK_START_AWS.md`  
**Experienced?** → Use `deploy-to-aws.sh`  
**Want automation?** → Use `cloudformation-template.yaml`

### **Step 2: Deploy**

#### **Option A: Manual (Recommended for first time)**
```bash
# Follow QUICK_START_AWS.md
# Time: 15 minutes
# Difficulty: Easy
```

#### **Option B: Automated Script**
```bash
# 1. Create EC2 instance in AWS Console
# 2. Run from your laptop:
chmod +x deploy-to-aws.sh
./deploy-to-aws.sh <EC2_IP> <SSH_KEY_PATH>

# Example:
./deploy-to-aws.sh 54.123.45.67 ~/.ssh/aflac-key.pem
```

#### **Option C: CloudFormation**
```bash
aws cloudformation create-stack \
  --stack-name aflac-assist \
  --template-body file://cloudformation-template.yaml \
  --parameters ParameterKey=KeyName,ParameterValue=your-key
```

### **Step 3: Access**
```
http://YOUR_EC2_IP:8000
```

---

## 💰 Cost Overview

| Configuration | Monthly Cost |
|---------------|--------------|
| **Basic** (t3.large) | ~$74 |
| **Reserved Instance** | ~$52 (save 30%) |
| **Spot Instance** | ~$22 (save 70%) |
| **Stopped** | ~$8 (storage only) |

---

## 📋 What You Need

### **AWS Requirements**
- ✅ AWS Account
- ✅ Credit card
- ✅ SSH key pair (we'll create this)

### **Technical Requirements**
- ✅ SSH client (built into Mac/Linux, PuTTY for Windows)
- ✅ Web browser
- ✅ 15 minutes of time

### **No Need For:**
- ❌ Docker knowledge
- ❌ Kubernetes experience
- ❌ DevOps background
- ❌ Programming skills

---

## 🏗️ Architecture

```
Internet
    ↓
EC2 Instance (t3.xlarge)
    ├─ FastAPI Application (Port 8000)
    ├─ Ollama Server (Port 11434)
    │   ├─ nomic-embed-text (embeddings)
    │   └─ mistral (text generation)
    ├─ ChromaDB (vector database)
    └─ EBS Storage (100 GB)
        ├─ PDF files
        └─ Embeddings
```

**Total Resources:** 1 EC2 instance + 1 EBS volume

---

## 📖 Documentation Guide

### **Start Here:**
1. **QUICK_START_AWS.md** ← Read this first!
   - 15-minute walkthrough
   - Step-by-step with examples
   - Perfect for beginners

### **Then Reference:**
2. **DEPLOYMENT_CHECKLIST.md**
   - Use as you deploy
   - Check off items as completed
   - Ensures nothing is missed

3. **AWS_DEPLOYMENT_GUIDE.md**
   - Detailed explanations
   - Multiple deployment methods
   - Security & optimization

4. **AWS_ARCHITECTURE_DIAGRAMS.md**
   - Visual diagrams
   - Data flow charts
   - Cost breakdowns

5. **DEPLOYMENT_SUMMARY.md**
   - Quick reference
   - File descriptions
   - Decision matrix

---

## 🔧 Deployment Methods Comparison

| Method | Time | Difficulty | Best For |
|--------|------|------------|----------|
| **Manual (Quick Start)** | 15 min | Easy | First deployment |
| **Automated Script** | 10 min | Medium | Repeated deployments |
| **CloudFormation** | 5 min | Advanced | Production/Teams |
| **Docker Compose** | 10 min | Medium | Container fans |

---

## ✅ What Gets Deployed

### **Software Installed:**
- ✅ Ubuntu 22.04 LTS
- ✅ Python 3.11+
- ✅ Ollama (with models)
- ✅ FastAPI application
- ✅ ChromaDB
- ✅ Systemd services (auto-start)

### **Configuration:**
- ✅ Security group (firewall)
- ✅ Storage (100 GB)
- ✅ Network (public IP)
- ✅ Services (auto-restart)

---

## 🎓 Learning Path

### **Beginner → Advanced**

#### **Level 1: Basic Deployment** (Week 1)
1. Read `QUICK_START_AWS.md`
2. Deploy to EC2
3. Upload PDFs
4. Test chat interface

#### **Level 2: Security** (Week 2)
1. Add HTTPS (Nginx + Let's Encrypt)
2. Configure authentication
3. Restrict security group
4. Set up backups

#### **Level 3: Optimization** (Week 3)
1. Monitor costs
2. Optimize instance size
3. Configure auto-scaling
4. Add CloudWatch monitoring

#### **Level 4: Production** (Week 4)
1. High availability setup
2. Load balancer
3. Multiple availability zones
4. Disaster recovery plan

---

## 🆘 Troubleshooting Quick Reference

### **Can't SSH to EC2**
```bash
# Check your IP
curl https://checkip.amazonaws.com

# Update security group to allow this IP
# In AWS Console: EC2 → Security Groups → Edit inbound rules
```

### **Application Not Loading**
```bash
# SSH into instance
ssh -i key.pem ubuntu@YOUR_IP

# Check status
sudo systemctl status aflac-assist
sudo systemctl status ollama

# View logs
sudo journalctl -u aflac-assist -f
```

### **Out of Memory**
```bash
# Upgrade instance in AWS Console:
# Stop instance → Actions → Instance Settings → Change Instance Type → t3.2xlarge
```

### **Ollama Models Missing**
```bash
# Re-pull models
ollama pull nomic-embed-text
ollama pull mistral
```

---

## 💡 Pro Tips

### **Save Money:**
- 💰 Stop instance when not in use (saves ~$120/month)
- 💰 Use Reserved Instance for production (save 30%)
- 💰 Set billing alerts ($150 threshold)

### **Improve Performance:**
- 🚀 Use g4dn.xlarge for GPU acceleration
- 🚀 Increase EBS IOPS for faster disk
- 🚀 Use smaller chunk sizes for faster queries

### **Enhance Security:**
- 🔒 Always restrict SSH to your IP
- 🔒 Enable HTTPS for production
- 🔒 Add authentication
- 🔒 Regular security updates

---

## 📞 Support Resources

### **Get Help:**
1. Check `DEPLOYMENT_CHECKLIST.md` troubleshooting section
2. Review application logs on EC2
3. Search AWS documentation
4. Create GitHub issue

### **Common Resources:**
- AWS Console: https://console.aws.amazon.com
- EC2 Dashboard: EC2 → Instances
- Billing: Billing Dashboard → Bills
- Support: AWS Support Center

---

## 🎯 Success Metrics

**You're successful when:**
- ✅ Can access application at `http://YOUR_IP:8000`
- ✅ Can upload PDFs
- ✅ Can ask questions and get answers
- ✅ Sources are cited correctly
- ✅ Cost is within budget

---

## 📊 Deployment Timeline

```
┌─────────────────────────────────────────────────┐
│ Deployment Timeline (First Time)                │
├─────────────────────────────────────────────────┤
│ Read documentation        │ 5 min               │
│ Create AWS account        │ 5 min (if needed)   │
│ Launch EC2 instance       │ 3 min               │
│ SSH connection            │ 2 min               │
│ Install Ollama            │ 3 min               │
│ Download models           │ 5 min               │
│ Install application       │ 2 min               │
│ Test & verify             │ 3 min               │
├─────────────────────────────────────────────────┤
│ TOTAL                     │ ~15 minutes         │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Update Instructions

### **Update Application Code:**
```bash
# SSH to instance
ssh -i key.pem ubuntu@YOUR_IP

# Pull latest changes
cd /opt/aflac-assist
git pull

# Restart application
sudo systemctl restart aflac-assist
```

### **Update Ollama:**
```bash
# Update Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Restart service
sudo systemctl restart ollama
```

---

## 🎉 Ready to Deploy?

### **Your Next Steps:**

1. **Open** `QUICK_START_AWS.md`
2. **Follow** the step-by-step guide
3. **Check off** items in `DEPLOYMENT_CHECKLIST.md`
4. **Access** your app at `http://YOUR_IP:8000`
5. **Celebrate** 🎊

---

## 📝 Files Created Summary

```
✅ Fixed: get_embedding_function.py (port 11434)
✅ Created: Dockerfile
✅ Created: docker-compose.yml
✅ Created: start.sh
✅ Created: ec2-user-data.sh
✅ Created: cloudformation-template.yaml
✅ Created: deploy-to-aws.sh
✅ Created: QUICK_START_AWS.md
✅ Created: AWS_DEPLOYMENT_GUIDE.md
✅ Created: AWS_ARCHITECTURE_DIAGRAMS.md
✅ Created: DEPLOYMENT_CHECKLIST.md
✅ Created: DEPLOYMENT_SUMMARY.md
✅ Created: README_DEPLOYMENT.md
✅ Updated: .gitignore
```

---

**Total Package:** 14 files  
**Documentation:** ~30,000 words  
**Deployment Time:** 15 minutes  
**Estimated Cost:** $134/month  

---

## 🚀 Let's Go!

**Start with:** `QUICK_START_AWS.md`

Good luck with your deployment! 🎯
