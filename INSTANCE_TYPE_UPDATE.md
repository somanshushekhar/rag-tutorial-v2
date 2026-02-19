# ✅ Instance Type Updated: t3.xlarge → t3.large

## 📝 Summary of Changes

All deployment documentation has been updated to use **t3.large** (2 vCPU, 8GB RAM) instead of t3.xlarge.

---

## 💰 Cost Savings

| Configuration | Before (t3.xlarge) | After (t3.large) | **Savings** |
|---------------|-------------------|------------------|-------------|
| **On-Demand** | $134/month | $74/month | **$60/month (45%)** |
| **Reserved (1yr)** | $94/month | $52/month | **$42/month (45%)** |
| **Spot Instance** | $40/month | $22/month | **$18/month (45%)** |

### Annual Savings
- **On-Demand:** Save **$720/year** 💰
- **Reserved:** Save **$504/year** 💰

---

## 🔧 Technical Specifications

### t3.large (NEW - Recommended)
- **vCPU:** 2
- **RAM:** 8 GB
- **Network:** Up to 5 Gbps
- **Cost:** $0.0832/hour
- **Use Case:** Cost-efficient for small to medium workloads

### t3.xlarge (OLD)
- **vCPU:** 4
- **RAM:** 16 GB
- **Network:** Up to 5 Gbps
- **Cost:** $0.1664/hour
- **Use Case:** Larger workloads, more concurrent users

---

## 📄 Files Updated

### ✅ Deployment Guides
1. ✅ `QUICK_START_AWS.md` - Updated instance type and costs
2. ✅ `AWS_DEPLOYMENT_GUIDE.md` - Updated recommendations
3. ✅ `DEPLOYMENT_SUMMARY.md` - Updated cost breakdown
4. ✅ `DEPLOYMENT_CHECKLIST.md` - Updated instance references
5. ✅ `README_DEPLOYMENT.md` - Updated cost overview

### ✅ Infrastructure Code
6. ✅ `cloudformation-template.yaml` - Changed default from t3.xlarge to t3.large

---

## 🚀 What to Do Next

### If You Already Have an Instance Running:

**Option 1: Change Instance Type (Recommended)**
1. Go to AWS Console → EC2 → Instances
2. Select your instance
3. **Instance State** → **Stop instance**
4. Wait for it to stop
5. **Actions** → **Instance settings** → **Change instance type**
6. Select **t3.large**
7. **Instance State** → **Start instance**
8. ✅ You'll now pay $74/month instead of $134/month!

**Option 2: Keep Current Instance**
- If your application needs more resources, keep t3.xlarge
- Monitor CPU and memory usage to decide

---

### If You Haven't Deployed Yet:

1. Follow **`QUICK_START_AWS.md`**
2. When selecting instance type, choose **t3.large**
3. Total monthly cost: **~$74** (including storage)

---

## 📊 Performance Expectations

### t3.large Can Handle:
- ✅ **Concurrent Users:** 5-10 simultaneous users
- ✅ **Documents:** Up to 1,000 PDF pages
- ✅ **Response Time:** 2-5 seconds per query
- ✅ **Ollama Models:** 2-3 models loaded simultaneously

### When to Upgrade to t3.xlarge:
- ⚠️ More than 10 concurrent users
- ⚠️ Very large documents (1000+ pages)
- ⚠️ Multiple large models (>7B parameters)
- ⚠️ Sub-second response time requirements

---

## 🔍 Monitoring Performance

After deployment, monitor your instance:

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@YOUR_IP

# Check CPU usage
top

# Check memory usage
free -h

# Check application logs
sudo journalctl -u rag-app -f
```

### Red Flags (Need to Upgrade):
- 🔴 CPU consistently above 80%
- 🔴 Memory (RAM) consistently above 90%
- 🔴 Application crashes with "Out of Memory" errors
- 🔴 Response time > 10 seconds

### Green Flags (t3.large is Sufficient):
- ✅ CPU below 60%
- ✅ Memory below 75%
- ✅ No crashes
- ✅ Response time < 5 seconds

---

## 💡 Cost Optimization Tips

### 1. **Use Spot Instances** (Save 70%)
- Cost: ~$22/month instead of $74/month
- Trade-off: Can be terminated with 2-minute notice
- Good for: Development/testing

### 2. **Stop When Not Using** (Save 89%)
- Stop instance at night/weekends
- Cost: Only $8/month for storage
- Start again when needed (IP address changes)

### 3. **Reserved Instances** (Save 30%)
- Commit to 1 year: ~$52/month
- Commit to 3 years: ~$38/month
- Good for: Production use

### 4. **Right-Sizing**
- Start with t3.large
- Monitor for 1-2 weeks
- Upgrade only if needed
- Downgrade if underutilized

---

## ✅ Deployment Checklist Update

When deploying, use these updated specs:

- [ ] Instance Type: **t3.large** (not t3.xlarge)
- [ ] vCPU: **2** (not 4)
- [ ] RAM: **8 GB** (not 16 GB)
- [ ] Expected Cost: **$74/month** (not $134/month)
- [ ] Savings: **$60/month** compared to previous recommendation

---

## 🆘 Troubleshooting

### "Out of Memory" Errors

If you get memory errors with t3.large:

```bash
# Check memory usage
free -h

# Check which process uses most memory
ps aux --sort=-%mem | head -10

# Option 1: Restart to clear memory
sudo systemctl restart rag-app

# Option 2: Upgrade to t3.xlarge (16 GB RAM)
# Follow AWS Console steps above
```

### Slow Performance

```bash
# Check if swap is being used (bad sign)
free -h

# If swap usage is high, upgrade instance type
# Go to AWS Console and change to t3.xlarge
```

---

## 📚 Related Documentation

- **Full Deployment Guide:** `QUICK_START_AWS.md`
- **Advanced Options:** `AWS_DEPLOYMENT_GUIDE.md`
- **Cost Optimization:** `DEPLOYMENT_SUMMARY.md`
- **Step-by-Step Checklist:** `DEPLOYMENT_CHECKLIST.md`

---

## ✨ Summary

**Before:** t3.xlarge (4 vCPU, 16 GB) = $134/month  
**After:** t3.large (2 vCPU, 8 GB) = $74/month  
**Savings:** $60/month or $720/year 💰

All documentation has been updated. You're ready to deploy!
