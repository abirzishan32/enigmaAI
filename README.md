# Private Sketch-to-Number with FHE

A privacy-preserving digit recognition application using **Fully Homomorphic Encryption (FHE)** with TenSEAL.

## Quick Start

### 1. Install Dependencies

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2. Train the Model

```bash
cd backend/digit-recognize
python3 train_model.py
```

Wait 5-10 minutes for training to complete.

### 3. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 4. Open Browser

Navigate to: **http://localhost:3000**

Draw a digit and click "Classify with FHE"!

---

## 📚 Full Documentation

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for complete details on:
- Architecture overview
- Training customization
- API documentation
- Production deployment
- Troubleshooting

---

## 🔧 Technology Stack

- **Backend:** Python, FastAPI, PyTorch, TenSEAL
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **FHE:** TenSEAL (CKKS scheme)

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── app.py                    # FastAPI server
│   ├── model.pth                 # Trained model (generated)
│   ├── full_context.bin          # FHE context (generated)
│   ├── requirements.txt
│   └── digit-recognize/
│       ├── train_model.py        # Training script
│       └── dataset/              # MNIST images
│           ├── training/
│           └── testing/
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   ├── components/
│   │   └── sections/
│   │       └── demo-fhe.tsx      # Main demo component
│   ├── lib/
│   │   └── fhe-client.ts         # FHE client library
│   └── package.json
└── IMPLEMENTATION_GUIDE.md       # Complete guide
```

---

## ✨ Features

✅ **Privacy-Preserving**: Server never sees your drawing in plain text  
✅ **FHE-based**: Uses TenSEAL for homomorphic encryption  
✅ **Real-time**: Draw and classify instantly  
✅ **Visualized**: See each step of the FHE pipeline  
✅ **Customizable**: Easy to modify model and training

---

## 🎯 How It Works

1. **User draws** digit on canvas
2. **Preprocessing** converts to 28×28 grayscale
3. **Encryption** (conceptual - TenSEAL is server-side)
4. **Server inference** on encrypted data
5. **Decryption** reveals prediction

The server performs inference without ever seeing your drawing!

---

## 🚀 Next Steps

### For Development:
- Experiment with different model architectures
- Try different FHE parameters
- Add more digits or datasets

### For Production:
- Use Concrete.js for true client-side FHE
- Remove clear inference endpoint
- Add authentication and rate limiting
- Deploy on cloud infrastructure

---

## 📖 Learn More

- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://www.microsoft.com/en-us/research/project/microsoft-seal/)
- [Concrete ML by Zama](https://docs.zama.ai/concrete-ml)

---

## 🤝 Support

Need help? Check:
1. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Complete documentation
2. [Troubleshooting section](IMPLEMENTATION_GUIDE.md#troubleshooting)
3. [API Documentation](IMPLEMENTATION_GUIDE.md#api-documentation)

---

**Built with ❤️ for Privacy-Preserving ML**
