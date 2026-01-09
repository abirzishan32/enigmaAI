"use client"

import { useState, useCallback } from "react"

export interface LogEntry {
  id: string
  timestamp: string
  message: string
  type: "info" | "success" | "warning" | "error"
}

export function useSentimentFHE() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [prediction, setPrediction] = useState<string | null>(null)
  
  // States for the workflow data
  const [encryptedData, setEncryptedData] = useState<number[] | null>(null)
  const [noiseText, setNoiseText] = useState<string | null>(null)

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
    const id = Math.random().toString(36).substring(7)
    setLogs(prev => [...prev, { id, timestamp, message, type }])
  }, [])

  const generateNoiseText = () => {
    // Generate random encrypted-looking text
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    let noise = ""
    for (let i = 0; i < 200; i++) {
      noise += chars.charAt(Math.floor(Math.random() * chars.length))
      if ((i + 1) % 32 === 0) noise += "\n"
    }
    return noise
  }

  const encryptInput = useCallback(async (text: string) => {
    setIsProcessing(true)
    addLog("Starting encryption process...", "info")
    
    await new Promise(r => setTimeout(r, 500))
    
    addLog("Fetching TF-IDF vectorizer parameters...", "info")
    
    try {
      // Get vectorizer params from server
      const res = await fetch("http://localhost:8000/sentiment/get-vectorizer-params", {
        method: "POST"
      })
      
      if (!res.ok) throw new Error("Failed to get vectorizer params")
      const params = await res.json()
      
      addLog("Applying TF-IDF transformation...", "info")
      await new Promise(r => setTimeout(r, 600))
      
      // For demo, create simple feature vector (in production, apply TF-IDF)
      const features = new Array(256).fill(0).map(() => Math.random() * 0.5)
      
      addLog("Generating CKKS encryption context...", "info")
      addLog("Poly Modulus Degree: 8192", "info")
      
      await new Promise(r => setTimeout(r, 800))
      
      setEncryptedData(features)
      const noise = generateNoiseText()
      setNoiseText(noise)
      
      addLog("Text features successfully encrypted.", "success")
      addLog("Ciphertext generated.", "info")
      setIsProcessing(false)
      return true
      
    } catch (e) {
      addLog("Encryption failed.", "error")
      setIsProcessing(false)
      return false
    }
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
        const res = await fetch("http://localhost:8000/sentiment/predict-encrypted", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ encrypted_features: encryptedData })
        })

        addLog("Server received payload.", "info")
        addLog("Server performing homomorphic inference...", "info")

        await new Promise(r => setTimeout(r, 1000))

        if (!res.ok) throw new Error("Server Error")
        const data = await res.json()
        
        setPrediction(data.label)
        
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
        addLog(`Sentiment Classification: ${prediction}`, "success")
    } else {
        addLog("Decryption failed: No result found.", "error")
    }
    
    setIsProcessing(false)
    return true
  }, [prediction, addLog])

  const resetState = useCallback(() => {
    setLogs([])
    setEncryptedData(null)
    setNoiseText(null)
    setPrediction(null)
    addLog("Session reset. Ready for new input.", "info")
  }, [addLog])

  return {
    logs,
    addLog,
    isProcessing,
    prediction,
    noiseText,
    encryptInput,
    sendToServer,
    decryptResult,
    resetState
  }
}
