# ✅ AWS Deployment Checklist

Use this checklist to ensure a successful deployment.

---

## 📋 Pre-Deployment Checklist

### **Local Preparation**
- [ ] All code committed to GitHub
- [ ] Port bug fixed (11434 not 11435) ✅ **DONE**
- [ ] `.gitignore` updated to exclude AWS keys ✅ **DONE**
- [ ] Test application works locally
- [ ] PDF files ready for upload

### **AWS Account Setup**
- [ ] AWS account created and verified
- [ ] Credit card added to AWS account
- [ ] Billing alerts configured ($150/month threshold)
- [ ] AWS CLI installed (optional but helpful)
- [ ] AWS region selected (e.g., us-east-1)

---

## 🚀 Deployment Checklist

### **Phase 1: EC2 Instance Setup**
- [ ] EC2 key pair created and downloaded (`aflac-assist-key.pem`)
- [ ] Key file saved securely
- [ ] Key file permissions set (`chmod 400`)
- [ ] EC2 instance launched (t3.large or larger)
- [ ] Security group configured:
  - [ ] Port 22 (SSH) - Your IP only
  - [ ] Port 80 (HTTP) - 0.0.0.0/0
  - [ ] Port 443 (HTTPS) - 0.0.0.0/0
  - [ ] Port 8000 (FastAPI) - 0.0.0.0/0
- [ ] Instance state: Running
- [ ] Public IP address noted

### **Phase 2: Initial Connection**
- [ ] Can SSH into instance successfully
- [ ] System updates completed (`sudo apt-get update`)
- [ ] No connection errors

### **Phase 3: Ollama Installation**
- [ ] Ollama installed
- [ ] Ollama service started
- [ ] Ollama service enabled (auto-start)
- [ ] Model `nomic-embed-text` downloaded
- [ ] Model `mistral` downloaded
- [ ] Can run: `ollama list` successfully

### **Phase 4: Application Installation**
- [ ] Git installed
- [ ] Python 3.11+ installed
- [ ] Repository cloned to `/opt/aflac-assist`
- [ ] Virtual environment created
- [ ] Python dependencies installed
- [ ] `data/` directory created
- [ ] `chroma/` directory created
- [ ] Correct file permissions set

### **Phase 5: Service Configuration**
- [ ] Systemd service file created (`aflac-assist.service`)
- [ ] Service daemon reloaded
- [ ] Service started successfully
- [ ] Service enabled (auto-start)
- [ ] Service status shows "active (running)"

### **Phase 6: Verification**
- [ ] Can access `http://<IP>:8000` in browser
- [ ] Upload page loads correctly
- [ ] No console errors in browser
- [ ] Application logs show no errors

---

## 📄 Post-Deployment Checklist

### **Data Setup**
- [ ] PDF files uploaded to `/opt/aflac-assist/data/`
- [ ] Documents ingested successfully
- [ ] ChromaDB populated
- [ ] Can verify chunks: Check chroma/ directory size

### **Functionality Testing**
- [ ] Can upload new PDF through web interface
- [ ] Upload triggers background processing
- [ ] Chat interface accessible
- [ ] Can ask questions
- [ ] Receives streaming responses
- [ ] Sources displayed correctly
- [ ] No timeout errors

### **Performance Testing**
- [ ] Response time < 5 seconds for queries
- [ ] No memory errors
- [ ] No disk space warnings
- [ ] CPU usage reasonable (<80%)

---

## 🔒 Security Checklist

### **Immediate Security**
- [ ] SSH restricted to specific IP (not 0.0.0.0/0)
- [ ] SSH key not shared publicly
- [ ] `.pem` file not in Git repository
- [ ] Strong EC2 password (if using password auth)

### **Production Security** (Optional but Recommended)
- [ ] HTTPS enabled (Nginx + Let's Encrypt)
- [ ] HTTP Basic Authentication added
- [ ] API rate limiting configured
- [ ] File upload validation added
- [ ] Firewall (UFW) configured
- [ ] CloudWatch monitoring enabled
- [ ] AWS CloudTrail enabled

---

## 💰 Cost Management Checklist

### **Cost Tracking**
- [ ] AWS Budget alert set up
- [ ] Cost Explorer enabled
- [ ] EC2 instance tagged (`Name: Aflac-Assist-Chat`)
- [ ] Billing dashboard reviewed

### **Cost Optimization**
- [ ] Consider Reserved Instance for 30% savings
- [ ] Stop instance when not in use
- [ ] Delete old EBS snapshots
- [ ] Monitor data transfer costs
- [ ] Review monthly bill

---

## 📊 Monitoring Checklist

### **Application Monitoring**
- [ ] Can view application logs: `sudo journalctl -u aflac-assist -f`
- [ ] Can view Ollama logs: `sudo journalctl -u ollama -f`
- [ ] No error messages in logs
- [ ] Disk space monitored: `df -h`
- [ ] Memory usage monitored: `free -h`

### **Uptime Monitoring** (Optional)
- [ ] CloudWatch alarms configured
- [ ] Health check endpoint added
- [ ] Uptime monitoring service (e.g., UptimeRobot)
- [ ] Email alerts configured

---

## 🔄 Backup Checklist

### **Data Backup**
- [ ] ChromaDB backup script created
- [ ] Backup cron job scheduled
- [ ] S3 bucket created for backups (optional)
- [ ] Backup restoration tested
- [ ] PDF files backed up separately

### **Instance Backup**
- [ ] AMI (snapshot) created
- [ ] EBS volume snapshot created
- [ ] Snapshot schedule configured
- [ ] Can restore from snapshot

---

## 📝 Documentation Checklist

### **Internal Documentation**
- [ ] EC2 instance IP documented
- [ ] SSH key location documented
- [ ] Admin credentials documented (if added auth)
- [ ] Deployment date recorded
- [ ] Architecture diagram saved
- [ ] Troubleshooting notes documented

### **User Documentation**
- [ ] User guide created for uploading PDFs
- [ ] Chat interface instructions provided
- [ ] Known limitations documented
- [ ] Support contact information provided

---

## 🆘 Troubleshooting Checklist

### **If Application Not Working**
- [ ] Check service status: `sudo systemctl status aflac-assist`
- [ ] Check Ollama status: `sudo systemctl status ollama`
- [ ] View recent logs: `sudo journalctl -u aflac-assist -n 50`
- [ ] Check port 8000 is listening: `sudo netstat -tulpn | grep 8000`
- [ ] Check security group allows port 8000
- [ ] Restart services if needed

### **If Slow Performance**
- [ ] Check CPU usage: `top`
- [ ] Check memory usage: `free -h`
- [ ] Check disk space: `df -h`
- [ ] Consider upgrading instance type
- [ ] Review application logs for errors

### **If Out of Disk Space**
- [ ] Clean old logs: `sudo journalctl --vacuum-time=7d`
- [ ] Remove unused Ollama models: `ollama rm <model>`
- [ ] Resize EBS volume if needed
- [ ] Move old PDFs to S3

---

## ✅ Final Verification

### **Complete Deployment Test**

Run this full workflow to verify everything works:

1. **Upload Test**
   - [ ] Navigate to `http://<IP>:8000`
   - [ ] Upload a small PDF (< 5 MB)
   - [ ] Wait for "Upload successful" message
   - [ ] Check logs for processing completion

2. **Chat Test**
   - [ ] Click "Go to Chat Interface"
   - [ ] Type a simple question
   - [ ] Receive streaming response
   - [ ] Sources displayed correctly
   - [ ] Response is relevant

3. **Performance Test**
   - [ ] Upload larger PDF (> 10 MB)
   - [ ] Process completes without errors
   - [ ] Ask complex question
   - [ ] Response time acceptable

4. **Persistence Test**
   - [ ] Ask same question again
   - [ ] Response is consistent
   - [ ] Restart application: `sudo systemctl restart aflac-assist`
   - [ ] Previous documents still available

---

## 📅 Maintenance Schedule

### **Daily**
- [ ] Check application is running
- [ ] Review logs for errors

### **Weekly**
- [ ] Review AWS costs
- [ ] Check disk space
- [ ] Review backup status

### **Monthly**
- [ ] Update system packages
- [ ] Update Python dependencies
- [ ] Test backup restoration
- [ ] Review security group rules
- [ ] Optimize costs

### **Quarterly**
- [ ] Review AWS Reserved Instance options
- [ ] Audit user access
- [ ] Test disaster recovery plan
- [ ] Update documentation

---

## 🎯 Success Criteria

**Your deployment is successful when:**

✅ All items in "Deployment Checklist" are checked  
✅ All items in "Functionality Testing" pass  
✅ No critical errors in logs  
✅ Users can upload PDFs and chat successfully  
✅ Response time is acceptable (< 5 seconds)  
✅ Cost is within budget (~$134/month)  

---

## 📞 Emergency Contacts

**In case of critical issues:**

### **AWS Support**
- Console: https://console.aws.amazon.com/support/
- Phone: Available with support plan

### **Technical Issues**
- Application Logs: `sudo journalctl -u aflac-assist -f`
- GitHub Issues: https://github.com/somanshushekhar/rag-tutorial-v2/issues

### **Cost Issues**
- AWS Billing: https://console.aws.amazon.com/billing/
- Stop instance immediately if costs spike

---

## 📊 Deployment Status Tracking

```
Deployment Date: _______________
EC2 Instance ID: _______________
Public IP: _______________
Instance Type: _______________
Monthly Cost: $_______________
Deployment Method: [ ] Manual [ ] Script [ ] CloudFormation
HTTPS Enabled: [ ] Yes [ ] No
Authentication: [ ] Yes [ ] No
Backup Configured: [ ] Yes [ ] No
Monitoring Enabled: [ ] Yes [ ] No
```

---

## 🎓 Additional Resources

- **AWS Documentation:** https://docs.aws.amazon.com/ec2/
- **Ollama Docs:** https://github.com/ollama/ollama
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **ChromaDB Docs:** https://docs.trychroma.com/

---

**Last Updated:** [Today's Date]  
**Deployment Engineer:** [Your Name]  
**Review Status:** [ ] Complete [ ] In Progress
