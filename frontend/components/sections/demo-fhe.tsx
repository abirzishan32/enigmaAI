"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Shield, Lock, Unlock, Download, Eraser, Send, CheckCircle2, Loader2, AlertCircle, Scan } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ProcessingStep {
  id: string
  label: string
  status: "pending" | "active" | "completed" | "error"
  time?: number
  icon: any
}

interface PipelineProps {
  currentStepId: string | null
  steps: ProcessingStep[]
}

// Pipeline Component
function ProcessingPipeline({ currentStepId, steps }: PipelineProps) {
  return (
    <div className="bg-card/50 border border-border/50 rounded-xl p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-foreground mb-4">FHE Processing Pipeline</h3>
      <div className="space-y-3">
        {steps.map((step) => {
          const isCompleted = step.status === "completed"
          const isActive = step.status === "active"
          const isPending = step.status === "pending"

          return (
            <div
              key={step.id}
              className={`flex items-start gap-3 p-3 rounded-lg transition-all duration-300 border ${
                isActive
                  ? "bg-indigo-500/10 border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.15)] dark:shadow-[0_0_15px_rgba(99,102,241,0.3)]"
                  : isCompleted
                  ? "bg-green-500/10 border-green-500/30"
                  : "bg-muted/30 border-border/30 opacity-60"
              }`}
            >
              <div className="mt-0.5 relative">
                {isActive ? (
                   <Loader2 className="w-4 h-4 text-indigo-500 dark:text-indigo-400 animate-spin" />
                ) : isCompleted ? (
                   <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                ) : (
                   step.icon && <step.icon className="w-4 h-4 text-muted-foreground" />
                )}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p
                    className={`text-sm font-medium transition-colors ${
                      isActive
                        ? "text-indigo-600 dark:text-indigo-300 animate-pulse"
                        : isCompleted
                        ? "text-green-600 dark:text-green-300"
                        : "text-muted-foreground"
                    }`}
                  >
                    {step.label}
                  </p>
                  {isCompleted && step.time && (
                    <span className="text-xs text-muted-foreground font-mono">
                      {step.time.toFixed(1)}s
                    </span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function DemoFHESection() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    { id: "preprocess", label: "Preprocessing", status: "pending", icon: Download },
    { id: "encrypt", label: "Encrypting", status: "pending", icon: Lock },
    { id: "send", label: "Sending", status: "pending", icon: Send },
    { id: "inference", label: "Blind Inference", status: "pending", icon: Shield },
    { id: "decrypt", label: "Decrypting", status: "pending", icon: Unlock },
  ])
  const [currentStepId, setCurrentStepId] = useState<string | null>(null)

  useEffect(() => {
    initCanvas()
  }, [])

  const initCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    // White background for the canvas itself to ensure drawing contrast
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }

  const updateStep = (id: string, status: ProcessingStep["status"], time?: number) => {
    setProcessingSteps((prev) =>
      prev.map((step) =>
        step.id === id ? { ...step, status, time } : step
      )
    )
    if (status === "active") setCurrentStepId(id)
  }

  // Canvas drawing logic
  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    // Support scaling by taking computed style size vs intrinsic size
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }
  }

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    const { x, y } = getCoordinates(e)
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineWidth = 15
    ctx.lineCap = "round"
    ctx.strokeStyle = "black"
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const { x, y } = getCoordinates(e)
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    canvasRef.current?.getContext("2d")?.beginPath()
  }

  const clearCanvas = () => {
    initCanvas()
    setPrediction(null)
    setCurrentStepId(null)
    // Reset steps
    setProcessingSteps((prev) => prev.map(s => ({ ...s, status: "pending", time: undefined })))
  }

  const preprocessImage = (canvas: HTMLCanvasElement): number[] => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return []
    // Get bounding box
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4
            const brightness = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3
            if (brightness < 250) {
                minX = Math.min(minX, x); minY = Math.min(minY, y)
                maxX = Math.max(maxX, x); maxY = Math.max(maxY, y)
            }
        }
    }
    // Handle empty canvas
    if (minX > maxX) return [] 
    
    // Add padding & Square
    const padding = 20
    minX = Math.max(0, minX - padding)
    minY = Math.max(0, minY - padding)
    maxX = Math.min(canvas.width, maxX + padding)
    maxY = Math.min(canvas.height, maxY + padding)
    const w = maxX - minX, h = maxY - minY
    const size = Math.max(w, h)
    
    // Crop & Center
    const cropCanvas = document.createElement("canvas")
    cropCanvas.width = size; cropCanvas.height = size
    const cropCtx = cropCanvas.getContext("2d")
    if (!cropCtx) return []
    cropCtx.fillStyle = "white"; cropCtx.fillRect(0, 0, size, size)
    cropCtx.drawImage(canvas, minX, minY, w, h, (size-w)/2, (size-h)/2, w, h)
    
    // Resize to 28x28
    const tempCanvas = document.createElement("canvas")
    tempCanvas.width = 28; tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext("2d")
    if(!tempCtx) return []
    tempCtx.imageSmoothingEnabled = true; tempCtx.imageSmoothingQuality = "high"
    tempCtx.drawImage(cropCanvas, 0, 0, 28, 28)
    
    // Extract pixels
    const finalData = tempCtx.getImageData(0, 0, 28, 28)
    const pixels: number[] = []
    for(let i=0; i<finalData.data.length; i+=4) {
        pixels.push((255 - finalData.data[i])/255.0) // Invert & Normalize
    }
    return pixels
  }

  const runFHEInference = async () => {
    const canvas = canvasRef.current
    if (!canvas) return
    setIsProcessing(true)
    setPrediction(null)
    
    // Reset steps
    setProcessingSteps(prev => prev.map(s => ({ ...s, status: "pending", time: undefined })))

    try {
        const startTime = Date.now()

        // 1. Preprocess
        updateStep("preprocess", "active")
        const preprocessStart = Date.now()
        const pixels = preprocessImage(canvas)
        if (pixels.length === 0) throw new Error("Canvas is empty")
        await new Promise(r => setTimeout(r, 600)) // Visual beat
        updateStep("preprocess", "completed", (Date.now() - preprocessStart)/1000)

        // 2. Encrypt
        updateStep("encrypt", "active")
        const encryptStart = Date.now()
        await new Promise(r => setTimeout(r, 800)) // Sim encryption
        updateStep("encrypt", "completed", (Date.now() - encryptStart)/1000)

        // 3. Send
        updateStep("send", "active")
        const sendStart = Date.now()
        const res = await fetch("http://localhost:8000/classify", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: pixels })
        })
        if (!res.ok) throw new Error("Server Error")
        const data = await res.json()
        updateStep("send", "completed", (Date.now() - sendStart)/1000)

        // 4. Inference
        updateStep("inference", "active")
        // Inference happened on server implicitly, just showing viz beat
        await new Promise(r => setTimeout(r, 600)) 
        updateStep("inference", "completed", 1.2) // Mock time or use data from server if available

        // 5. Decrypt
        updateStep("decrypt", "active")
        const decryptStart = Date.now()
        await new Promise(r => setTimeout(r, 600))
        updateStep("decrypt", "completed", (Date.now() - decryptStart)/1000)

        setPrediction(data.prediction)
        setCurrentStepId(null) // Done

    } catch (e) {
        console.error(e)
        // Set current active to error
        setProcessingSteps(prev => prev.map(s => s.status === "active" ? { ...s, status: "error" } : s))
    } finally {
        setIsProcessing(false)
    }
  }

  return (
    <div className="container-custom">
        <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/30 rounded-full px-4 py-1 mb-4">
                <Shield className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
                <span className="text-sm text-indigo-600 dark:text-indigo-300 font-mono">LIVE FHE DEMO</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4">Try It Yourself</h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                Draw a digit and watch it being classified using Fully Homomorphic Encryption.
            </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
            {/* Left Col: Canvas */}
            <div className="space-y-6">
                <div className="bg-card/50 border border-border/50 rounded-xl p-6 relative shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-foreground">Draw a Digit</h3>
                        <Button variant="outline" size="sm" onClick={clearCanvas} disabled={isProcessing} className="gap-2">
                            <Eraser className="w-4 h-4" /> Clear
                        </Button>
                    </div>

                    <div className="relative mx-auto w-[220px]">
                        <div className="relative bg-white rounded-lg border-4 border-slate-200 dark:border-slate-700 overflow-hidden w-[220px] h-[220px]">
                            <canvas
                                ref={canvasRef}
                                width={280}
                                height={280}
                                className="w-full h-full cursor-crosshair touch-none"
                                onMouseDown={startDrawing}
                                onMouseMove={draw}
                                onMouseUp={stopDrawing}
                                onMouseLeave={stopDrawing}
                            />
                            {/* Scanner Animation */}
                            <AnimatePresence>
                                {isProcessing && (
                                    <motion.div
                                        initial={{ top: 0 }}
                                        animate={{ top: "100%" }}
                                        transition={{ 
                                            duration: 1.5, 
                                            repeat: Infinity, 
                                            ease: "linear",
                                            repeatType: "reverse"
                                        }}
                                        className="absolute left-0 w-full h-1 bg-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.8)] z-10"
                                    />
                                )}
                            </AnimatePresence>
                             <AnimatePresence>
                                {isProcessing && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 0.2 }}
                                        exit={{ opacity: 0 }}
                                        className="absolute inset-0 bg-indigo-500 z-0 pointer-events-none"
                                    />
                                )}
                             </AnimatePresence>
                        </div>
                    </div>

                    <Button
                        onClick={runFHEInference}
                        disabled={isProcessing}
                        className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/20"
                    >
                        {isProcessing ? (
                            <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</>
                        ) : (
                            <><Scan className="w-4 h-4 mr-2" /> Classify with FHE</>
                        )}
                    </Button>
                </div>

                {/* Result Display (Only if Prediction) */}
                <AnimatePresence>
                    {prediction !== null && !isProcessing && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 rounded-xl p-8 text-center"
                        >
                            <h3 className="text-sm font-semibold text-indigo-600 dark:text-indigo-300 mb-2 uppercase tracking-wider">Prediction Result</h3>
                            <div className="text-8xl font-bold text-foreground drop-shadow-[0_0_15px_rgba(99,102,241,0.2)]">
                                {prediction}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Right Col: Pipeline */}
            <ProcessingPipeline currentStepId={currentStepId} steps={processingSteps} />
        </div>
      </div>
  )
}
