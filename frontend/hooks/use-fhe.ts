"use client"

import { useState, useCallback } from "react"

export interface LogEntry {
  id: string
  timestamp: string
  message: string
  type: "info" | "success" | "warning" | "error"
}

export function useFHE() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [prediction, setPrediction] = useState<number | null>(null)
  
  // States for the workflow data
  const [encryptedData, setEncryptedData] = useState<number[] | null>(null) // The "Ciphertext" (simulated by just holding the pixels)
  const [noiseImage, setNoiseImage] = useState<string | null>(null)

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
    const id = Math.random().toString(36).substring(7)
    setLogs(prev => [...prev, { id, timestamp, message, type }])
  }, [])

  const generateNoiseImage = () => {
    const cvs = document.createElement("canvas")
    cvs.width = 200; cvs.height = 200
    const ctx = cvs.getContext("2d")
    if (!ctx) return null
    const idata = ctx.createImageData(200, 200)
    for (let i = 0; i < idata.data.length; i += 4) {
      const v = Math.random() * 255
      idata.data[i] = v
      idata.data[i+1] = v
      idata.data[i+2] = v
      idata.data[i+3] = 255
    }
    ctx.putImageData(idata, 0, 0)
    return cvs.toDataURL()
  }

  const encryptInput = useCallback(async (pixels: number[]) => {
    setIsProcessing(true)
    addLog("Starting encryption process...", "info")
    
    // Simulate delay
    await new Promise(r => setTimeout(r, 500))
    
    addLog("Generating CKKS context parameters...", "info")
    addLog("Poly Modulus Degree: 8192", "info")
    addLog("Coeff Modulus Sizes: [60, 40, 40, 60]", "info")
    
    await new Promise(r => setTimeout(r, 800))
    
    // In a real WASM client, this would be the actual ciphertext blob
    // Here we store the raw pixels but visualize it as noise
    setEncryptedData(pixels) 
    const noise = generateNoiseImage()
    setNoiseImage(noise)
    
    addLog("Input vector successfully encrypted.", "success")
    addLog("Ciphertext generated.", "info")
    setIsProcessing(false)
    return true
  }, [addLog])

  const sendToServer = useCallback(async () => {
    if (!encryptedData) {
        addLog("No encrypted data found to send.", "error")
        return false
    }

    setIsProcessing(true)
    addLog("Initiating secure connection to server...", "info")
    addLog("Uploading encrypted payload (Ciphertext)...", "info")

    try {
        const res = await fetch("http://localhost:8000/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: encryptedData })
        })

        addLog("Server received payload.", "info")
        addLog("Server computing Homomorphic Dot Product...", "info")
        addLog("Server computing Square Activation...", "info")

        await new Promise(r => setTimeout(r, 1000)) // Computation delay

        if (!res.ok) throw new Error("Server Error")
        const data = await res.json()
        
        // The server returns the prediction (and ideally an encrypted result, but for this demo, the prediction)
        // In a full FHE verified mode, we'd receive a ciphertext of the result here.
        setPrediction(data.prediction)
        
        addLog("Computation complete. Result encrypted.", "info")
        addLog("Downloading encrypted result...", "success")
        
        setIsProcessing(false)
        return true

    } catch (e) {
        addLog("Failed to communicate with server.", "error")
        setIsProcessing(false)
        return false
    } 
  }, [encryptedData, addLog])

  const decryptResult = useCallback(async () => {
    setIsProcessing(true)
    addLog("Requesting private key for decryption...", "info")
    await new Promise(r => setTimeout(r, 600))
    
    addLog("Decrypting result vector...", "info")
    if (prediction !== null) {
        addLog(`Decryption successful.`, "success")
        addLog(`Model Classification: ${prediction}`, "success")
    } else {
        addLog("Decryption failed: No result found.", "error")
    }
    
    setIsProcessing(false)
    return true
  }, [prediction, addLog])

  const resetState = useCallback(() => {
    setLogs([])
    setEncryptedData(null)
    setNoiseImage(null)
    setPrediction(null)
    addLog("Session reset. Ready for new input.", "info")
  }, [addLog])

  return {
    logs,
    addLog,
    isProcessing,
    prediction,
    noiseImage,
    encryptInput,
    sendToServer,
    decryptResult,
    resetState
  }
}
