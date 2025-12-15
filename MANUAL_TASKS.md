# Manual Tasks and Questions for P04 ResNet Wafer Map Classifier

## Overview
This document lists tasks that require manual intervention or clarification before full implementation can proceed.

---

## Critical Inputs Needed

### 1. Data Availability
**Question:** Do you have access to real wafer map data or should we use synthetic/public datasets?

**Options:**
- **Option A:** Use real proprietary wafer map data from semiconductor manufacturing
  - Requires access to STDF files or wafer map images
  - Need annotation labels or annotators available
  
- **Option B:** Use public WM-811K dataset (811,457 wafer maps)
  - Available at: http://mirlab.org/dataset/public/
  - Pre-labeled with 9 defect patterns
  - Good for proof-of-concept and baseline

- **Option C:** Generate synthetic wafer maps for demonstration
  - Create simulated defect patterns programmatically
  - Faster development but less realistic

**Recommendation:** Start with Option B (WM-811K) for initial implementation, then migrate to real data.

**Status:** ⏳ AWAITING INPUT

---

### 2. Cloud vs On-Premise Deployment
**Question:** Where should this system be deployed?

**Options:**
- **AWS/Azure/GCP:** Full cloud deployment (easier scalability, $15K/month)
- **On-Premise:** Private data center (data security, higher initial cost)
- **Hybrid:** Development in cloud, production on-premise

**Implications:**
- Cloud: Use managed services (RDS, ElastiCache, EKS)
- On-Premise: Self-managed PostgreSQL, Redis, Kubernetes

**Status:** ⏳ AWAITING INPUT

---

### 3. GPU Availability
**Question:** Do you have access to GPUs for training?

**Requirements:**
- **Training:** 4× NVIDIA V100 (32GB) or A100 (40GB) preferred
- **Inference:** Optional T4 for faster predictions (can use CPU)

**Options:**
- **Cloud GPUs:** AWS p3.8xlarge ($12.24/hour), Azure NC24s_v3
- **On-Premise GPUs:** Purchase/lease workstations with GPUs
- **Alternatives:** Use Google Colab Pro+ for training ($50/month, limited hours)

**Status:** ⏳ AWAITING INPUT

---

### 4. Annotation Resources
**Question:** Who will annotate the wafer maps?

**Requirements:**
- Phase 1: 1,000 labeled wafers (baseline)
- Active Learning: +500 wafers across 5 iterations
- Time: ~10 minutes per wafer

**Options:**
- **Internal FA Engineers:** 2-3 engineers, 2-3 weeks part-time
- **External Annotators:** Hire contractors, provide training
- **Use Pre-Labeled Dataset:** WM-811K already labeled

**Status:** ⏳ AWAITING INPUT

---

### 5. Integration Requirements
**Question:** Does this need to integrate with existing systems?

**Potential Integrations:**
- **Test Systems:** STDF parser for automatic wafer map ingestion
- **Yield Analysis Tools:** Export predictions to existing dashboards
- **FA Tools:** SEM/TEM coordinate export for physical analysis
- **Authentication:** Corporate SSO (Azure AD, Okta)

**Status:** ⏳ AWAITING INPUT

---

## Implementation Decisions

### 6. Database Choice
**Current Plan:** PostgreSQL 16+ for primary database

**Alternatives:**
- MongoDB (document store, more flexible schema)
- TimescaleDB (time-series optimization for metrics)
- Keep PostgreSQL (stable, ACID, well-documented)

**Status:** ✅ DECIDED - PostgreSQL (can change if needed)

---

### 7. Frontend Framework
**Current Plan:** React 18 + TypeScript + Next.js 14

**Alternatives:**
- Vue.js 3 (lighter, easier learning curve)
- Streamlit (Python-based, very fast prototyping)
- Keep React (industry standard, rich ecosystem)

**Status:** ✅ DECIDED - React (most versatile for complex UIs)

---

### 8. Model Deployment Strategy
**Current Plan:** Start with CPU inference (ONNX), add GPU if needed

**Reasoning:**
- CPU inference <2s meets requirements for most use cases
- Lower infrastructure cost ($800/month vs $8,000/month for GPUs)
- Can upgrade to GPU easily if latency becomes bottleneck

**Status:** ✅ DECIDED - CPU-first approach

---

## Reduced Scope Options (If Timeline/Budget Constraints)

### MVP (Minimum Viable Product) - 6 Weeks Instead of 12
**Includes:**
1. ✅ Supervised baseline model (ResNet-50 U-Net, 1K labeled wafers)
2. ✅ Basic inference API (FastAPI, ONNX)
3. ✅ Simple web UI for predictions (React, basic visualization)
4. ✅ Docker containers for easy deployment
5. ✅ Basic monitoring (Prometheus, Grafana)

**Excludes:**
- ❌ Active learning (manual annotation instead)
- ❌ Semi-supervised learning (supervised-only, may need more data)
- ❌ Advanced UI features (annotation tool, training monitor)
- ❌ Kubernetes deployment (Docker Compose sufficient for pilot)
- ❌ Production hardening (comprehensive testing, security audits)

**Trade-offs:**
- Lower IoU (93% vs 95.8%)
- More annotation cost ($166K vs $25K)
- Faster time-to-value (6 weeks vs 12 weeks)

**Status:** ⏳ AWAITING INPUT - Full implementation or MVP?

---

## Current Assumptions (Confirm or Correct)

1. **Programming Language:** Python 3.11+ for backend/ML ✅
2. **Development OS:** macOS (based on workspace path) ✅
3. **Team Size:** Solo developer or small team (<5 people) ✅
4. **Timeline:** 12 weeks full implementation acceptable ✅
5. **Budget:** ~$15K/month infrastructure + development time ⏳
6. **Use Case:** Production semiconductor manufacturing environment ⏳
7. **Scale:** 10,000 wafers/day processing target ⏳
8. **Security:** Enterprise-grade (RBAC, encryption, audit logs) ⏳

**Status:** ⏳ AWAITING CONFIRMATION

---

## Next Steps

### Immediate Actions (After Inputs Received)
1. **Download WM-811K Dataset** (if using public data)
   - 811,457 wafer maps (300×300 PNG images)
   - 9 defect classes pre-labeled
   - ~8GB compressed download

2. **Set Up Development Environment**
   - Python 3.11 virtual environment
   - Install PyTorch, FastAPI, React dependencies
   - Configure Docker Desktop

3. **Create Project Structure**
   - Backend (FastAPI services)
   - ML (model training, inference)
   - Frontend (React applications)
   - Infrastructure (Docker, Kubernetes, monitoring)

4. **Baseline Model Training**
   - Load WM-811K dataset
   - Implement ResNet-50 U-Net
   - Train on 1,000 labeled samples
   - Target: >92% IoU

### Progress Tracking
I will update this document as decisions are made and tasks are completed.

**Last Updated:** December 10, 2025

---

## How to Provide Input

Please reply with answers to the critical questions above, for example:

```
1. Data: Use WM-811K public dataset for now
2. Deployment: AWS cloud deployment
3. GPU: Use AWS p3.8xlarge on-demand for training
4. Annotation: Use WM-811K pre-labeled data (no manual annotation)
5. Integration: Standalone system for now, add integrations later
6. Scope: Full 12-week implementation preferred
```

I'll proceed with reasonable defaults where answers aren't critical to initial development, but the above inputs will significantly affect the implementation approach.
