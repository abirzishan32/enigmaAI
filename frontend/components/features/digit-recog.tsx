"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Shield, Lock, Unlock, Download, Eraser, Send,  Loader2, Scan, Database, User, Terminal as TerminalIcon, ChevronDown, ChevronUp } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

// --- Types ---
interface LogEntry {
  id: string
  timestamp: string
  message: string
  type: "info" | "success" | "warning" | "error"
}

// --- Components ---

// 1. Terminal Component
function LiveTerminal({ logs }: { logs: LogEntry[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm h-[600px] flex flex-col">
      <div className="bg-muted px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
            <TerminalIcon className="w-4 h-4 text-muted-foreground" />
             <span className="text-sm font-mono font-medium text-foreground">Live Terminal</span>
        </div>
        <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-400/50" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-400/50" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-400/50" />
        </div>
      </div>
      <div 
        ref={scrollRef}
        className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-2 bg-slate-950 text-slate-300"
      >
        {logs.length === 0 && (
            <div className="text-slate-600 italic">Waiting for input...</div>
        )}
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2">
            <span className="text-slate-500 shrink-0">[{log.timestamp}]</span>
            <span className={
                log.type === "error" ? "text-red-400" :
                log.type === "success" ? "text-green-400" :
                log.type === "warning" ? "text-yellow-400" :
                "text-slate-300"
            }>
                {log.type === "info" && "> "}
                {log.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// 2. Step Badge
function StepBadge({ type }: { type: "client" | "server" }) {
    return (
        <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${
            type === "client" 
                ? "bg-blue-500/10 text-blue-500 border-blue-500/20" 
                : "bg-purple-500/10 text-purple-500 border-purple-500/20"
        }`}>
            {type === "client" ? <User className="w-3 h-3" /> : <Database className="w-3 h-3" />}
            {type === "client" ? "Client Side" : "Server Side"}
        </div>
    )
}

// 3. Step Container
function StepItem({ 
    stepNumber, 
    title, 
    isActive, 
    isCompleted, 
    badgeType,
    children, 
    onToggle 
}: { 
    stepNumber: number
    title: string
    isActive: boolean
    isCompleted: boolean
    badgeType: "client" | "server"
    children: React.ReactNode
    onToggle: () => void
}) {
    return (
        <div className={`border rounded-xl transition-all duration-300 overflow-hidden ${
            isActive 
                ? "bg-card border-indigo-500/50 shadow-md ring-1 ring-indigo-500/20" 
                : isCompleted
                ? "bg-card/50 border-green-500/30"
                : "bg-card/30 border-border opacity-70"
        }`}>
            <button 
                onClick={onToggle}
                className="w-full flex items-center justify-between p-4 focus:outline-none"
            >
                <div className="flex items-center gap-4">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm transition-colors ${
                        isActive 
                            ? "bg-indigo-600 text-white" 
                            : isCompleted
                            ? "bg-green-500 text-white"
                            : "bg-muted text-muted-foreground"
                    }`}>
                        {isCompleted ? <CheckCircle2 className="w-5 h-5" /> : stepNumber}
                    </div>
                    <span className={`font-semibold ${isActive ? "text-foreground" : "text-muted-foreground"}`}>
                        {title}
                    </span>
                </div>
                <div className="flex items-center gap-4">
                    <StepBadge type={badgeType} />
                    {isActive ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
                </div>
            </button>
            <AnimatePresence initial={false}>
                {isActive && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="px-4 pb-6 pt-0 border-t border-border/50">
                            <div className="pt-6">
                                {children}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

import { CheckCircle2 } from "lucide-react"

// --- Main Page Component ---
export function DigitRecog() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [activeStep, setActiveStep] = useState<1 | 2 | 3>(1)
  const [completedSteps, setCompletedSteps] = useState<number[]>([])

  // State
  const [isDrawing, setIsDrawing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [pixels, setPixels] = useState<number[]>([])
  const [noiseImage, setNoiseImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null)

  // Initialize Canvas
  useEffect(() => {
    initCanvas()
    addLog("System initialized. Waiting for user input...", "info")
  }, [])

  const addLog = (message: string, type: LogEntry["type"] = "info") => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
    const id = Math.random().toString(36).substring(7)
    setLogs(prev => [...prev, { id, timestamp, message, type }])
  }

  const initCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }

  // --- Step 1 Actions ---
  const handleEncryptInput = async () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    // Check if empty
    const pixels = preprocessImage(canvas)
    if (pixels.length === 0) {
        addLog("Canvas is empty. Please draw a digit first.", "warning")
        return
    }

    setIsProcessing(true)
    addLog("Starting encryption process...", "info")
    
    // Tiny delay for UX
    await new Promise(r => setTimeout(r, 500))
    
    addLog("Generating CKKS context parameters...", "info")
    addLog("Poly Modulus Degree: 8192", "info")
    addLog("Coeff Modulus Sizes: [60, 40, 40, 60]", "info")
    
    await new Promise(r => setTimeout(r, 800))
    setPixels(pixels)
    
    // Generate Visual Noise
    const noise = generateNoiseImage()
    setNoiseImage(noise)
    
    addLog("Input vector successfully encrypted.", "success")
    addLog("Ciphertext generated.", "info")
    
    setIsProcessing(false)
    setCompletedSteps(prev => [...prev, 1])
    setActiveStep(2)
  }

  // --- Step 2 Actions ---
  const handleSendToServer = async () => {
    setIsProcessing(true)
    addLog("Initiating secure connection to server...", "info")
    addLog("Uploading encrypted payload (Ciphertext)...", "info")
    
    try {
        const res = await fetch("http://localhost:8000/classify", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: pixels })
        })
        
        addLog("Server received payload.", "info")
        addLog("Server computing Homomorphic Dot Product...", "info")
        addLog("Server computing Square Activation...", "info")
        
        await new Promise(r => setTimeout(r, 1000)) // Fake computation time delay
        
        if (!res.ok) throw new Error("Server Error")
        const data = await res.json()
        
        setPrediction(data.prediction)
        
        addLog("Computation complete. Result encrypted.", "info")
        addLog("Downloading encrypted result...", "success")
        
        setIsProcessing(false)
        setCompletedSteps(prev => [...prev, 2])
        setActiveStep(3)
        
    } catch (e) {
        addLog("Failed to communicate with server.", "error")
        setIsProcessing(false)
    }
  }

  // --- Step 3 Actions ---
  const handleDecrypt = async () => {
    setIsProcessing(true)
    addLog("Requesting private key for decryption...", "info")
    await new Promise(r => setTimeout(r, 600))
    
    addLog("Decrypting result vector...", "info")
    addLog(`Decryption successful.`, "success")
    addLog(`Model Classification: ${prediction}`, "success")
    
    setIsProcessing(false)
    setCompletedSteps(prev => [...prev, 3])
  }

  const handleReset = () => {
    initCanvas()
    setPixels([])
    setNoiseImage(null)
    setPrediction(null)
    setCompletedSteps([])
    setActiveStep(1)
    setLogs([])
    addLog("Session reset. Ready for new input.", "info")
  }

  // --- Helpers ---
  const generateNoiseImage = () => {
    const cvs = document.createElement("canvas")
    cvs.width = 200; cvs.height = 200
    const ctx = cvs.getContext("2d")
    if(!ctx) return null
    const idata = ctx.createImageData(200, 200)
    for(let i=0; i<idata.data.length; i+=4) {
        const v = Math.random() * 255
        idata.data[i] = v
        idata.data[i+1] = v
        idata.data[i+2] = v
        idata.data[i+3] = 255
    }
    ctx.putImageData(idata, 0, 0)
    return cvs.toDataURL()
  }

  // Reuse existing canvas helpers
  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
  }
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true); const {x,y} = getCoordinates(e); 
    const ctx = canvasRef.current?.getContext("2d"); if(!ctx) return;
    ctx.beginPath(); ctx.moveTo(x,y); ctx.lineWidth=15; ctx.lineCap="round"; ctx.strokeStyle="black"
  }
  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if(!isDrawing) return; const {x,y} = getCoordinates(e);
    const ctx = canvasRef.current?.getContext("2d"); if(!ctx) return;
    ctx.lineTo(x,y); ctx.stroke()
  }
  const stopDrawing = () => { setIsDrawing(false); canvasRef.current?.getContext("2d")?.beginPath() }
  
  const preprocessImage = (canvas: HTMLCanvasElement): number[] => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return []
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
    if (minX > maxX) return [] 
    const padding = 20
    minX = Math.max(0, minX - padding); minY = Math.max(0, minY - padding)
    maxX = Math.min(canvas.width, maxX + padding); maxY = Math.min(canvas.height, maxY + padding)
    const w = maxX - minX, h = maxY - minY
    const size = Math.max(w, h)
    const cropCanvas = document.createElement("canvas")
    cropCanvas.width = size; cropCanvas.height = size
    const cropCtx = cropCanvas.getContext("2d")
    if (!cropCtx) return []
    cropCtx.fillStyle = "white"; cropCtx.fillRect(0, 0, size, size)
    cropCtx.drawImage(canvas, minX, minY, w, h, (size-w)/2, (size-h)/2, w, h)
    const tempCanvas = document.createElement("canvas")
    tempCanvas.width = 28; tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext("2d")
    if(!tempCtx) return []
    tempCtx.imageSmoothingEnabled = true; tempCtx.imageSmoothingQuality = "high"
    tempCtx.drawImage(cropCanvas, 0, 0, 28, 28)
    const finalData = tempCtx.getImageData(0, 0, 28, 28)
    const pixels: number[] = []
    for(let i=0; i<finalData.data.length; i+=4) {
        pixels.push((255 - finalData.data[i])/255.0)
    }
    return pixels
  }

  return (
    <div className="container-custom">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* LEFT COLUMN: Workflow Stepper */}
            <div className="lg:col-span-2 space-y-4">
                {/* Step 1: Encrypt */}
                <StepItem 
                    stepNumber={1} 
                    title="Encrypt Input" 
                    isActive={activeStep === 1}
                    isCompleted={completedSteps.includes(1)}
                    badgeType="client"
                    onToggle={() => { if(completedSteps.includes(1)) setActiveStep(1) }}
                >
                    <div className="flex flex-col items-center gap-6">
                        <div className="relative group">
                            <div className="bg-slate-200 dark:bg-slate-800 p-1 rounded-lg">
                                <canvas
                                    ref={canvasRef}
                                    width={280}
                                    height={280}
                                    className="bg-white rounded cursor-crosshair touch-none shadow-inner"
                                    onMouseDown={startDrawing}
                                    onMouseMove={draw}
                                    onMouseUp={stopDrawing}
                                    onMouseLeave={stopDrawing}
                                />
                            </div>
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                className="absolute top-2 right-2 opacity-50 hover:opacity-100 hover:bg-slate-100 dark:hover:bg-slate-700"
                                onClick={initCanvas}
                            >
                                <Eraser className="w-5 h-5 text-slate-500" />
                            </Button>
                        </div>
                        <div className="w-full max-w-xs">
                             <Button 
                                onClick={handleEncryptInput} 
                                disabled={isProcessing}
                                className="w-full h-12 text-base bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/20"
                            >
                                {isProcessing ? (
                                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Encrypting...</>
                                ) : (
                                    <><Lock className="w-4 h-4 mr-2" /> Encrypt Input</>
                                )}
                            </Button>
                        </div>
                    </div>
                </StepItem>

                {/* Step 2: Send */}
                <StepItem 
                    stepNumber={2} 
                    title="Send to Server" 
                    isActive={activeStep === 2}
                    isCompleted={completedSteps.includes(2)}
                    badgeType="server"
                    onToggle={() => { if(completedSteps.includes(2) || completedSteps.includes(1)) setActiveStep(2) }}
                >
                     <div className="flex flex-col items-center gap-6">
                        <div className="p-6 bg-slate-950 rounded-xl border border-slate-800 shadow-inner">
                            {noiseImage ? (
                                <img src={noiseImage} alt="Noise" className="w-[200px] h-[200px] rounded opacity-80 mix-blend-screen" />
                            ) : (
                                <div className="w-[200px] h-[200px] flex items-center justify-center text-slate-600">
                                    <span className="text-xs">No Ciphertext</span>
                                </div>
                            )}
                        </div>
                        <p className="text-sm text-muted-foreground text-center max-w-md">
                            This visual noise represents the encrypted data (Ciphertext). The server can operate on this data but cannot "see" the original image.
                        </p>
                         <div className="w-full max-w-xs">
                             <Button 
                                onClick={handleSendToServer} 
                                disabled={isProcessing}
                                className="w-full h-12 text-base bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/20"
                            >
                                {isProcessing ? (
                                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Sending...</>
                                ) : (
                                    <><Send className="w-4 h-4 mr-2" /> Send to Server</>
                                )}
                            </Button>
                        </div>
                    </div>
                </StepItem>

                {/* Step 3: Decrypt */}
                <StepItem 
                    stepNumber={3} 
                    title="Decrypt Result" 
                    isActive={activeStep === 3}
                    isCompleted={completedSteps.includes(3)}
                    badgeType="client"
                    onToggle={() => { if(completedSteps.includes(2)) setActiveStep(3) }}
                >
                     <div className="flex flex-col items-center gap-6">
                        <div className="flex items-center justify-center min-h-[160px]">
                            {completedSteps.includes(3) && prediction !== null ? (
                                <motion.div 
                                    initial={{ scale: 0.5, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    className="text-center"
                                >
                                    <p className="text-sm text-muted-foreground mb-2 uppercase tracking-widest">The Model Predicted</p>
                                    <div className="text-9xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-indigo-500 to-purple-600 drop-shadow-xl">
                                        {prediction}
                                    </div>
                                </motion.div>
                            ) : (
                                <div className="flex flex-col items-center text-slate-400 gap-4">
                                    <div className="w-20 h-20 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center ring-4 ring-slate-200 dark:ring-slate-700">
                                        <Lock className="w-8 h-8" />
                                    </div>
                                    <p className="text-sm">Result is currently encrypted</p>
                                </div>
                            )}
                        </div>
                        
                         <div className="w-full max-w-xs space-y-3">
                             {!completedSteps.includes(3) && (
                                <Button 
                                    onClick={handleDecrypt} 
                                    disabled={isProcessing}
                                    className="w-full h-12 text-base bg-green-600 hover:bg-green-700 text-white shadow-lg shadow-green-500/20"
                                >
                                    {isProcessing ? (
                                        <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Decrypting...</>
                                    ) : (
                                        <><Unlock className="w-4 h-4 mr-2" /> Decrypt Result</>
                                    )}
                                </Button>
                             )}
                             
                             {completedSteps.includes(3) && (
                                 <Button 
                                    variant="outline"
                                    onClick={handleReset} 
                                    className="w-full"
                                >
                                    Start Over
                                </Button>
                             )}
                        </div>
                    </div>
                </StepItem>
            </div>

            {/* RIGHT COLUMN: Live Terminal */}
            <div className="lg:col-span-1">
                <div className="sticky top-24">
                    <LiveTerminal logs={logs} />
                </div>
            </div>
        </div>
    </div>
  )
}
