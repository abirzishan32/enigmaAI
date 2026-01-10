"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Shield, Lock, Unlock, Download, Eraser, Send,  Loader2, Scan, Database, User, Terminal as TerminalIcon, ChevronDown, ChevronUp, CheckCircle2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { useCanvas } from "@/hooks/use-canvas"
import { useFHE, LogEntry } from "@/hooks/use-fhe"

// --- Sub-Components ---

function LiveTerminal({ logs }: { logs: LogEntry[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm h-[400px] flex flex-col">
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
          <div key={log.id} className="flex gap-2 animate-in fade-in slide-in-from-left-1 duration-300">
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
                className="w-full flex items-center justify-between p-4 focus:outline-none hover:bg-muted/50 transition-colors"
                disabled={!isActive && !isCompleted}
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

// --- Main Component ---

export function DigitRecog() {
  const [activeStep, setActiveStep] = useState<1 | 2 | 3>(1)
  const [completedSteps, setCompletedSteps] = useState<number[]>([])

  // Hooks
  const { canvasRef, initCanvas, startDrawing, draw, stopDrawing, getPixelData } = useCanvas()
  const { logs, addLog, isProcessing, prediction, noiseImage, encryptInput, sendToServer, decryptResult, resetState } = useFHE()

  // Initialize
  useEffect(() => {
    addLog("System initialized. Waiting for user input...", "info")
  }, [addLog])

  // Handlers
  const onEncrypt = async () => {
    const pixels = getPixelData()
    if (pixels.length === 0) {
        addLog("Canvas is empty. Please draw a digit first.", "warning")
        return
    }
    const success = await encryptInput(pixels)
    if (success) {
        setCompletedSteps(prev => [...prev, 1])
        setActiveStep(2)
    }
  }

  const onSend = async () => {
    const success = await sendToServer()
    if (success) {
        setCompletedSteps(prev => [...prev, 2])
        setActiveStep(3)
    }
  }

  const onDecrypt = async () => {
    const success = await decryptResult()
    if (success) {
        setCompletedSteps(prev => [...prev, 3])
    }
  }

  const onReset = () => {
    resetState()
    initCanvas()
    setCompletedSteps([])
    setActiveStep(1)
  }

  return (
    <div className="container-custom h-full flex items-center py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full">
            
            {/* LEFT COLUMN: Workflow Stepper */}
            <div className="lg:col-span-2 space-y-3 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2">
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
                                    width={240}
                                    height={240}
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
                                onClick={onEncrypt} 
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
                     <div className="flex flex-col items-center gap-4">
                        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 shadow-inner">
                            {noiseImage ? (
                                <img src={noiseImage} alt="Noise" className="w-[160px] h-[160px] rounded opacity-80 mix-blend-screen" />
                            ) : (
                                <div className="w-[160px] h-[160px] flex items-center justify-center text-slate-600">
                                    <span className="text-xs">No Ciphertext</span>
                                </div>
                            )}
                        </div>
                        <p className="text-sm text-muted-foreground text-center max-w-md">
                            This visual noise represents the encrypted data (Ciphertext). The server can operate on this data but cannot "see" the original image.
                        </p>
                         <div className="w-full max-w-xs">
                             <Button 
                                onClick={onSend} 
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
                     <div className="flex flex-col items-center gap-4">
                        <div className="flex items-center justify-center min-h-[120px]">
                            {completedSteps.includes(3) && prediction !== null ? (
                                <motion.div 
                                    initial={{ scale: 0.5, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    className="text-center"
                                >
                                    <p className="text-xs text-muted-foreground mb-1 uppercase tracking-widest">The Model Predicted</p>
                                    <div className="text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-indigo-500 to-purple-600 drop-shadow-xl">
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
                                    onClick={onDecrypt} 
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
                                    onClick={onReset} 
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
