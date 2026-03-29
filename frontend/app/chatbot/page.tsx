"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp, Shield, Lock, Eye, EyeOff, Bot, Sparkles, BarChart3, AlertTriangle, CheckCircle, Shuffle, MapPin, ChevronDown, ChevronUp, Zap, Database, Brain, ArrowRight, RefreshCw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

interface EntityInfo {
  text: string;
  type: string;
  start: number;
  end: number;
  source: string;
  sensitivity: {
    identity_risk: number;
    query_necessity: number;
    reidentification_risk: number;
    combined_score: number;
    strategy: string;
  };
}

interface TransformationInfo {
  original: string;
  transformed: string;
  strategy: string;
  entity_type: string;
  reversible: boolean;
}

interface MetricsInfo {
  entities_detected: number;
  entities_redacted: number;
  entities_generalized: number;
  entities_preserved: number;
  entities_geo_obfuscated: number;
  privacy_score: number;
  utility_score: number;
  tradeoff_score: number;
}

interface IntentInfo {
  type: string;
  confidence: number;
  query_keywords: string[];
  disclosure_keywords: string[];
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  metadata?: {
    original?: string;
    redacted?: string;
    pii_map?: Record<string, string>;
    preserved_items?: Record<string, string>;
    raw_response?: string;
    intent?: IntentInfo;
    entities?: EntityInfo[];
    transformations?: TransformationInfo[];
    metrics?: MetricsInfo;
  };
}

export default function SecureChatbotPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/chat/secure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.content }),
      });

      const data = await res.json();
      
      // Debug: Log the API response
      console.log("API Response:", data);
      
      if (data.error) throw new Error(data.error);

      // Add Assistant Response
      const botMsg: ChatMessage = {
        role: "assistant",
        content: data.llm_response_restored,
        metadata: {
          original: data.original_prompt,
          redacted: data.redacted_prompt,
          pii_map: data.pii_map,
          preserved_items: data.preserved_items,
          raw_response: data.llm_response_raw,
          intent: data.intent,
          entities: data.entities,
          transformations: data.transformations,
          metrics: data.metrics
        }
      };
      
      setMessages((prev) => [...prev, botMsg]);

    } catch (e: any) {
        setMessages(prev => [...prev, { 
            role: "assistant", 
            content: "Sorry, I encountered an error connecting to the Secure Privacy Layer." 
        }]);
        console.error(e);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="h-[calc(100vh-65px)] overflow-hidden bg-background text-foreground font-sans flex flex-col transition-colors duration-300">

      {/* Main Chat Area */}
      <main className="flex-1 w-full flex flex-col overflow-hidden">
        
        {/* Scrollable Content Area - scrollbar at edge */}
        <div className="flex-1 overflow-y-auto chat-scroll" ref={scrollRef}>
          <div className="px-3 sm:px-6 lg:px-8 py-3 space-y-4">

        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center text-center space-y-4 opacity-50 min-h-[60vh]">
            <Shield className="w-16 h-16 text-muted-foreground" />
            <p className="text-muted-foreground max-w-sm">
              Your messages are intelligently analyzed and adaptively protected. 
              Query-critical entities are preserved while identity-critical data is transformed.
            </p>
            <div className="flex gap-2 text-xs">
              <span className="px-2 py-1 bg-red-500/10 text-red-400 rounded border border-red-500/20">Full Redaction</span>
              <span className="px-2 py-1 bg-amber-500/10 text-amber-400 rounded border border-amber-500/20">Generalization</span>
              <span className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded border border-blue-500/20">Preservation</span>
              <span className="px-2 py-1 bg-purple-500/10 text-purple-400 rounded border border-purple-500/20">Geo-Obfuscation</span>
            </div>
          </div>
        )}

          {messages.map((msg, idx) => (
            <MessageItem key={idx} msg={msg} />
          ))}
          {isLoading && (
             <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20">
                    <Bot className="w-4 h-4 text-primary" />
                </div>
                <div className="bg-card border border-border rounded-2xl rounded-tl-none p-4 w-fit">
                    <div className="flex gap-1">
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce delay-0" />
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce delay-150" />
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce delay-300" />
                    </div>
                </div>
             </div>
          )}
          </div>
        </div>

        {/* Input Area */}
        <div className="w-full px-3 sm:px-6 lg:px-8 pb-3 shrink-0">
        <div className="bg-card border border-input rounded-2xl p-2 relative shadow-2xl">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
                if(e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                }
            }}
            placeholder="Try: 'My name is Alice and I live in Dhaka, tell me about Paris'"
            className="bg-transparent border-none focus-visible:ring-0 min-h-[50px] max-h-[150px] resize-none pr-12 text-base text-card-foreground placeholder:text-muted-foreground"
          />
          <Button 
            size="icon" 
            onClick={handleSubmit} 
            disabled={isLoading || !input.trim()}
            className="absolute right-2 bottom-2 rounded-xl w-10 h-10 transition-all hover:scale-105"
          >
            <ArrowUp className="w-5 h-5" />
          </Button>
        </div>
        </div>
        
      </main>
    </div>
  );
}

function MessageItem({ msg }: { msg: ChatMessage }) {
    const isUser = msg.role === "user";
    const [showDebug, setShowDebug] = useState(false);
    
    return (
        <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center border shrink-0 ${
                isUser 
                ? "bg-secondary border-secondary" 
                : "bg-primary/10 border-primary/20"
            }`}>
               {isUser ? <div className="w-4 h-4 bg-secondary-foreground/50 rounded-full" /> : <Bot className="w-4 h-4 text-primary" />}
            </div>

            <div className={`flex flex-col gap-2 max-w-[95%] ${isUser ? "items-end" : "items-start w-full"}`}>
                <div className={`text-sm max-w-none ${
                    isUser 
                    ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-none px-3 py-2.5 leading-relaxed [&>p]:mb-2 [&>p:last-child]:mb-0 [&>ul]:list-disc [&>ul]:pl-4" 
                    : "text-foreground px-0 py-2 leading-8 w-full prose dark:prose-invert"
                    }`}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>

                {/* Privacy Inspector (Assistant Only) - Always show if metadata exists */}
                {!isUser && msg.metadata && (
                    <div className="w-full space-y-3">
                        {/* Always Visible: Pipeline Summary Card */}
                        <PipelineSummaryCard metadata={msg.metadata} />
                        
                        {/* Expandable: Detailed Analysis */}
                        {msg.metadata.entities && msg.metadata.entities.length > 0 && (
                        <button 
                             onClick={() => setShowDebug(!showDebug)}
                             className="text-xs flex items-center gap-1.5 text-primary hover:text-primary/80 transition-colors bg-primary/5 px-3 py-1.5 rounded-lg border border-primary/10"
                        >
                            {showDebug ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                            {showDebug ? "Hide Detailed Analysis" : "Show Detailed Pipeline Analysis"}
                        </button>
                        )}
                        
                        <AnimatePresence>
                            {showDebug && (
                                <motion.div 
                                    initial={{ height: 0, opacity: 0 }} 
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="overflow-hidden mt-2"
                                >
                                    <div className="bg-card rounded-xl border border-primary/20 p-4 text-xs font-mono space-y-4 shadow-inner">
                                        
                                        {/* Intent Classification */}
                                        {msg.metadata.intent && (
                                            <div className="space-y-2">
                                                <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold flex items-center gap-2">
                                                    <Sparkles className="w-3 h-3" />
                                                    Intent Classification
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    <IntentBadge type={msg.metadata.intent.type} confidence={msg.metadata.intent.confidence} />
                                                    {msg.metadata.intent.query_keywords.length > 0 && (
                                                        <div className="text-[10px] text-muted-foreground">
                                                            Query: {msg.metadata.intent.query_keywords.join(", ")}
                                                        </div>
                                                    )}
                                                    {msg.metadata.intent.disclosure_keywords.length > 0 && (
                                                        <div className="text-[10px] text-muted-foreground">
                                                            Disclosure: {msg.metadata.intent.disclosure_keywords.join(", ")}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        {/* Privacy Metrics */}
                                        {msg.metadata.metrics && (
                                            <div className="space-y-2">
                                                <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold flex items-center gap-2">
                                                    <BarChart3 className="w-3 h-3" />
                                                    Privacy-Utility Tradeoff
                                                </div>
                                                <div className="grid grid-cols-3 gap-2">
                                                    <MetricCard label="Privacy" value={msg.metadata.metrics.privacy_score} color="emerald" />
                                                    <MetricCard label="Utility" value={msg.metadata.metrics.utility_score} color="blue" />
                                                    <MetricCard label="Tradeoff" value={msg.metadata.metrics.tradeoff_score} color="purple" />
                                                </div>
                                                <div className="flex gap-2 text-[10px] text-muted-foreground">
                                                    <span>Detected: {msg.metadata.metrics.entities_detected}</span>
                                                    <span>•</span>
                                                    <span className="text-red-400">Redacted: {msg.metadata.metrics.entities_redacted}</span>
                                                    <span>•</span>
                                                    <span className="text-amber-400">Generalized: {msg.metadata.metrics.entities_generalized}</span>
                                                    <span>•</span>
                                                    <span className="text-blue-400">Preserved: {msg.metadata.metrics.entities_preserved}</span>
                                                    <span>•</span>
                                                    <span className="text-purple-400">Geo-Obf: {msg.metadata.metrics.entities_geo_obfuscated}</span>
                                                </div>
                                            </div>
                                        )}

                                        {/* Original vs Transformed */}
                                        <div className="space-y-1">
                                            <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold">Original Input</div>
                                            <div className="p-2 bg-destructive/10 border border-destructive/20 rounded text-destructive break-words">
                                                {msg.metadata.original}
                                            </div>
                                        </div>

                                        <div className="flex justify-center text-muted-foreground">
                                            <div className="bg-muted p-1 rounded-full border border-border">
                                                <Lock className="w-3 h-3" />
                                            </div>
                                        </div>

                                        <div className="space-y-1">
                                            <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold flex justify-between">
                                                <span>Sent to LLM</span>
                                                <span className="text-emerald-500">Adaptive Transformation</span>
                                            </div>
                                            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded text-emerald-600 dark:text-emerald-400 break-words">
                                                <RedactedText 
                                                    text={msg.metadata.redacted || ""} 
                                                    transformations={msg.metadata.transformations}
                                                />
                                            </div>
                                        </div>
                                        
                                        {/* Entity Sensitivity Analysis */}
                                        {msg.metadata.entities && msg.metadata.entities.length > 0 && (
                                            <div className="pt-2 border-t border-border">
                                                <div className="text-muted-foreground mb-2 uppercase tracking-widest text-[10px] font-bold">
                                                    Entity Sensitivity Analysis
                                                </div>
                                                <div className="space-y-2">
                                                    {msg.metadata.entities.map((entity, i) => (
                                                        <EntityCard key={i} entity={entity} transformation={msg.metadata?.transformations?.[i]} />
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                )}
            </div>
        </div>
    )
}

function IntentBadge({ type, confidence }: { type: string; confidence: number }) {
    const config: Record<string, { bg: string; border: string; text: string; icon: any; label: string }> = {
        query: { bg: "bg-blue-500/10", border: "border-blue-500/20", text: "text-blue-400", icon: CheckCircle, label: "Query" },
        disclosure: { bg: "bg-red-500/10", border: "border-red-500/20", text: "text-red-400", icon: AlertTriangle, label: "Disclosure" },
        hybrid: { bg: "bg-amber-500/10", border: "border-amber-500/20", text: "text-amber-400", icon: Shuffle, label: "Hybrid" }
    };
    
    const { bg, border, text, icon: Icon, label } = config[type] || config.query;
    
    return (
        <div className={`flex items-center gap-1.5 ${bg} ${text} px-2 py-1 rounded border ${border}`}>
            <Icon className="w-3 h-3" />
            <span className="font-semibold">{label}</span>
            <span className="opacity-60">({(confidence * 100).toFixed(0)}%)</span>
        </div>
    );
}

// ============================================================================
// PIPELINE SUMMARY CARD - Always visible after response
// ============================================================================

function PipelineSummaryCard({ metadata }: { metadata: ChatMessage["metadata"] }) {
    // Debug log
    console.log("PipelineSummaryCard metadata:", metadata);
    
    if (!metadata) return null;
    
    const { intent, metrics, transformations, entities, original, redacted, pii_map, preserved_items } = metadata;
    
    // Show card even with minimal data (just original/redacted)
    const hasAnyData = original || redacted || entities?.length || transformations?.length || pii_map;
    if (!hasAnyData) return null;
    
    // Group transformations by strategy
    const shuffled = transformations?.filter(t => t.strategy === "A" && t.transformed !== t.original) || [];
    const generalized = transformations?.filter(t => t.strategy === "B" && t.transformed !== t.original) || [];
    const preserved = transformations?.filter(t => t.strategy === "C") || [];
    const geoObfuscated = transformations?.filter(t => t.strategy === "D" && t.transformed !== t.original) || [];
    
    // Fallback: use pii_map if transformations not available
    const piiEntries = pii_map ? Object.entries(pii_map) : [];
    const preservedEntries = preserved_items ? Object.entries(preserved_items) : [];
    
    const hasTransformations = shuffled.length > 0 || generalized.length > 0 || preserved.length > 0 || geoObfuscated.length > 0;
    const hasPiiMap = piiEntries.length > 0 || preservedEntries.length > 0;
    
    const strategyNames: Record<string, string> = {
        A: "Full Redaction",
        B: "Generalization", 
        C: "Preservation",
        D: "Geo-Obfuscation"
    };

    return (
        <div className="w-full bg-gradient-to-br from-card to-card/50 rounded-xl border border-border p-4 space-y-4">
            {/* Header with Pipeline Flow */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-primary/10 rounded-lg">
                        <Zap className="w-4 h-4 text-primary" />
                    </div>
                    <span className="font-semibold text-sm">Privacy Pipeline Results</span>
                </div>
                {metrics && (
                    <div className="flex items-center gap-3 text-xs">
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-emerald-500" />
                            <span className="text-muted-foreground">Privacy:</span>
                            <span className="font-bold text-emerald-400">{(metrics.privacy_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-blue-500" />
                            <span className="text-muted-foreground">Utility:</span>
                            <span className="font-bold text-blue-400">{(metrics.utility_score * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Pipeline Flow Visualization */}
            <div className="flex items-center justify-between text-[10px] text-muted-foreground bg-muted/30 rounded-lg p-2">
                <div className="flex items-center gap-1">
                    <Database className="w-3 h-3" />
                    <span>Input</span>
                </div>
                <ArrowRight className="w-3 h-3" />
                <div className="flex items-center gap-1">
                    <Brain className="w-3 h-3" />
                    <span>NER</span>
                </div>
                <ArrowRight className="w-3 h-3" />
                <div className="flex items-center gap-1">
                    <Sparkles className="w-3 h-3" />
                    <span>Intent</span>
                </div>
                <ArrowRight className="w-3 h-3" />
                <div className="flex items-center gap-1">
                    <BarChart3 className="w-3 h-3" />
                    <span>Score</span>
                </div>
                <ArrowRight className="w-3 h-3" />
                <div className="flex items-center gap-1">
                    <RefreshCw className="w-3 h-3" />
                    <span>Transform</span>
                </div>
                <ArrowRight className="w-3 h-3" />
                <div className="flex items-center gap-1">
                    <Shield className="w-3 h-3 text-emerald-500" />
                    <span className="text-emerald-500">Safe Output</span>
                </div>
            </div>

            {/* Intent Classification */}
            {intent && (
                <div className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground">Intent Detected:</span>
                    <IntentBadge type={intent.type} confidence={intent.confidence} />
                </div>
            )}

            {/* Transformation Summary Grid */}
            {(hasTransformations || hasPiiMap) && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {/* Shuffled/Redacted */}
                <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-2">
                    <div className="flex items-center gap-1.5 mb-1">
                        <Lock className="w-3 h-3 text-red-400" />
                        <span className="text-[10px] font-semibold text-red-400 uppercase">Redacted</span>
                    </div>
                    <div className="text-lg font-bold text-red-400">
                        {shuffled.length > 0 ? shuffled.length : piiEntries.length}
                    </div>
                    {shuffled.length > 0 ? (
                        <div className="mt-1 space-y-0.5">
                            {shuffled.slice(0, 3).map((t, i) => (
                                <div key={i} className="text-[9px] text-red-300/70 truncate">
                                    {t.original} → {t.transformed}
                                </div>
                            ))}
                            {shuffled.length > 3 && (
                                <div className="text-[9px] text-red-300/50">+{shuffled.length - 3} more</div>
                            )}
                        </div>
                    ) : piiEntries.length > 0 && (
                        <div className="mt-1 space-y-0.5">
                            {piiEntries.slice(0, 3).map(([fake, original], i) => (
                                <div key={i} className="text-[9px] text-red-300/70 truncate">
                                    {original} → {fake}
                                </div>
                            ))}
                            {piiEntries.length > 3 && (
                                <div className="text-[9px] text-red-300/50">+{piiEntries.length - 3} more</div>
                            )}
                        </div>
                    )}
                </div>

                {/* Generalized */}
                <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-2">
                    <div className="flex items-center gap-1.5 mb-1">
                        <Shuffle className="w-3 h-3 text-amber-400" />
                        <span className="text-[10px] font-semibold text-amber-400 uppercase">Generalized</span>
                    </div>
                    <div className="text-lg font-bold text-amber-400">{generalized.length}</div>
                    {generalized.length > 0 && (
                        <div className="mt-1 space-y-0.5">
                            {generalized.slice(0, 3).map((t, i) => (
                                <div key={i} className="text-[9px] text-amber-300/70 truncate">
                                    {t.original} → {t.transformed}
                                </div>
                            ))}
                            {generalized.length > 3 && (
                                <div className="text-[9px] text-amber-300/50">+{generalized.length - 3} more</div>
                            )}
                        </div>
                    )}
                </div>

                {/* Preserved */}
                <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-2">
                    <div className="flex items-center gap-1.5 mb-1">
                        <CheckCircle className="w-3 h-3 text-blue-400" />
                        <span className="text-[10px] font-semibold text-blue-400 uppercase">Preserved</span>
                    </div>
                    <div className="text-lg font-bold text-blue-400">
                        {preserved.length > 0 ? preserved.length : preservedEntries.length}
                    </div>
                    {preserved.length > 0 ? (
                        <div className="mt-1 space-y-0.5">
                            {preserved.slice(0, 3).map((t, i) => (
                                <div key={i} className="text-[9px] text-blue-300/70 truncate">
                                    {t.original} (query-critical)
                                </div>
                            ))}
                            {preserved.length > 3 && (
                                <div className="text-[9px] text-blue-300/50">+{preserved.length - 3} more</div>
                            )}
                        </div>
                    ) : preservedEntries.length > 0 && (
                        <div className="mt-1 space-y-0.5">
                            {preservedEntries.slice(0, 3).map(([text, label], i) => (
                                <div key={i} className="text-[9px] text-blue-300/70 truncate">
                                    {text} ({label})
                                </div>
                            ))}
                            {preservedEntries.length > 3 && (
                                <div className="text-[9px] text-blue-300/50">+{preservedEntries.length - 3} more</div>
                            )}
                        </div>
                    )}
                </div>

                {/* Geo-Obfuscated */}
                <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-2">
                    <div className="flex items-center gap-1.5 mb-1">
                        <MapPin className="w-3 h-3 text-purple-400" />
                        <span className="text-[10px] font-semibold text-purple-400 uppercase">Geo-Obfuscated</span>
                    </div>
                    <div className="text-lg font-bold text-purple-400">{geoObfuscated.length}</div>
                    {geoObfuscated.length > 0 && (
                        <div className="mt-1 space-y-0.5">
                            {geoObfuscated.slice(0, 3).map((t, i) => (
                                <div key={i} className="text-[9px] text-purple-300/70 truncate">
                                    {t.original} → {t.transformed}
                                </div>
                            ))}
                            {geoObfuscated.length > 3 && (
                                <div className="text-[9px] text-purple-300/50">+{geoObfuscated.length - 3} more</div>
                            )}
                        </div>
                    )}
                </div>
            </div>
            )}

            {/* Fallback: Simple PII Map display if no detailed transformations */}
            {!hasTransformations && hasPiiMap && (
                <div className="bg-muted/30 rounded-lg p-3">
                    <div className="text-[10px] font-semibold text-muted-foreground uppercase mb-2">
                        Entity Transformations
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        {piiEntries.map(([fake, original], i) => (
                            <div key={i} className="flex items-center justify-between bg-red-500/10 p-1.5 rounded border border-red-500/20 text-[10px]">
                                <span className="text-red-400 truncate">{original}</span>
                                <span className="text-muted-foreground mx-1">→</span>
                                <span className="text-emerald-400 truncate">{fake}</span>
                            </div>
                        ))}
                        {preservedEntries.map(([text, label], i) => (
                            <div key={`p-${i}`} className="flex items-center justify-between bg-blue-500/10 p-1.5 rounded border border-blue-500/20 text-[10px]">
                                <span className="text-blue-400 truncate">{text}</span>
                                <span className="text-muted-foreground text-[9px]">(preserved)</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Scoring System Summary */}
            {entities && entities.length > 0 && (
                <div className="bg-muted/30 rounded-lg p-3">
                    <div className="text-[10px] font-semibold text-muted-foreground uppercase mb-2 flex items-center gap-1">
                        <BarChart3 className="w-3 h-3" />
                        Entity Scoring Summary
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-[10px]">
                            <thead>
                                <tr className="text-muted-foreground border-b border-border">
                                    <th className="text-left py-1 pr-2">Entity</th>
                                    <th className="text-left py-1 pr-2">Type</th>
                                    <th className="text-center py-1 pr-2">ID Risk</th>
                                    <th className="text-center py-1 pr-2">Query Need</th>
                                    <th className="text-center py-1 pr-2">Re-ID Risk</th>
                                    <th className="text-left py-1">Strategy</th>
                                    <th className="text-left py-1">Result</th>
                                </tr>
                            </thead>
                            <tbody>
                                {entities.map((entity, i) => {
                                    const transformation = metadata.transformations?.[i];
                                    const strategyConfig: Record<string, { color: string; label: string }> = {
                                        A: { color: "red", label: "Redact" },
                                        B: { color: "amber", label: "Generalize" },
                                        C: { color: "blue", label: "Preserve" },
                                        D: { color: "purple", label: "Geo-Obf" }
                                    };
                                    const config = strategyConfig[entity.sensitivity.strategy] || strategyConfig.A;
                                    
                                    return (
                                        <tr key={i} className="border-b border-border/50">
                                            <td className="py-1.5 pr-2 font-medium">{entity.text}</td>
                                            <td className="py-1.5 pr-2 text-muted-foreground">{entity.type}</td>
                                            <td className="py-1.5 pr-2 text-center">
                                                <span className={entity.sensitivity.identity_risk > 0.7 ? "text-red-400 font-bold" : "text-emerald-400"}>
                                                    {(entity.sensitivity.identity_risk * 100).toFixed(0)}%
                                                </span>
                                            </td>
                                            <td className="py-1.5 pr-2 text-center">
                                                <span className={entity.sensitivity.query_necessity > 0.6 ? "text-blue-400 font-bold" : "text-muted-foreground"}>
                                                    {(entity.sensitivity.query_necessity * 100).toFixed(0)}%
                                                </span>
                                            </td>
                                            <td className="py-1.5 pr-2 text-center">
                                                <span className={entity.sensitivity.reidentification_risk > 0.5 ? "text-amber-400" : "text-emerald-400"}>
                                                    {(entity.sensitivity.reidentification_risk * 100).toFixed(0)}%
                                                </span>
                                            </td>
                                            <td className={`py-1.5 pr-2 text-${config.color}-400 font-semibold`}>
                                                {config.label}
                                            </td>
                                            <td className="py-1.5">
                                                {transformation && transformation.transformed !== transformation.original ? (
                                                    <span className={`text-${config.color}-400`}>{transformation.transformed}</span>
                                                ) : (
                                                    <span className="text-blue-400">(kept)</span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Input/Output Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-2">
                    <div className="text-[10px] font-semibold text-red-400 uppercase mb-1 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" />
                        Original (Sensitive)
                    </div>
                    <div className="text-xs text-red-300/80 break-words font-mono">
                        {metadata.original}
                    </div>
                </div>
                <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-2">
                    <div className="text-[10px] font-semibold text-emerald-400 uppercase mb-1 flex items-center gap-1">
                        <Shield className="w-3 h-3" />
                        Sent to LLM (Protected)
                    </div>
                    <div className="text-xs text-emerald-300/80 break-words font-mono">
                        {metadata.redacted}
                    </div>
                </div>
            </div>
        </div>
    );
}

function MetricCard({ label, value, color }: { label: string; value: number; color: string }) {
    const percentage = (value * 100).toFixed(0);
    
    const colorClasses: Record<string, { bg: string; border: string; text: string; bar: string }> = {
        emerald: { bg: "bg-emerald-500/5", border: "border-emerald-500/20", text: "text-emerald-400", bar: "bg-emerald-500" },
        blue: { bg: "bg-blue-500/5", border: "border-blue-500/20", text: "text-blue-400", bar: "bg-blue-500" },
        purple: { bg: "bg-purple-500/5", border: "border-purple-500/20", text: "text-purple-400", bar: "bg-purple-500" },
        red: { bg: "bg-red-500/5", border: "border-red-500/20", text: "text-red-400", bar: "bg-red-500" },
        amber: { bg: "bg-amber-500/5", border: "border-amber-500/20", text: "text-amber-400", bar: "bg-amber-500" }
    };
    
    const classes = colorClasses[color] || colorClasses.emerald;
    
    return (
        <div className={`p-2 rounded border ${classes.bg} ${classes.border}`}>
            <div className={`${classes.text} text-lg font-bold`}>{percentage}%</div>
            <div className="text-muted-foreground text-[10px]">{label}</div>
            <div className="mt-1 h-1 bg-muted rounded-full overflow-hidden">
                <div className={`h-full ${classes.bar}`} style={{ width: `${percentage}%` }} />
            </div>
        </div>
    );
}

function EntityCard({ entity, transformation }: { entity: EntityInfo; transformation?: TransformationInfo }) {
    const strategyConfig: Record<string, { bg: string; border: string; text: string; label: string; icon: any }> = {
        A: { bg: "bg-red-500/5", border: "border-red-500/20", text: "text-red-400", label: "Full Redaction", icon: Lock },
        B: { bg: "bg-amber-500/5", border: "border-amber-500/20", text: "text-amber-400", label: "Generalization", icon: Shuffle },
        C: { bg: "bg-blue-500/5", border: "border-blue-500/20", text: "text-blue-400", label: "Preservation", icon: CheckCircle },
        D: { bg: "bg-purple-500/5", border: "border-purple-500/20", text: "text-purple-400", label: "Geo-Obfuscation", icon: MapPin }
    };
    
    const config = strategyConfig[entity.sensitivity.strategy] || strategyConfig.A;
    const IconComponent = config.icon;
    
    return (
        <div className={`p-2 rounded border ${config.bg} ${config.border}`}>
            <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                    <span className={`${config.text} font-semibold`}>{entity.text}</span>
                    <span className="text-[9px] uppercase opacity-50 bg-muted px-1 rounded">{entity.type}</span>
                </div>
                <div className={`flex items-center gap-1 ${config.text} text-[10px]`}>
                    <IconComponent className="w-3 h-3" />
                    {config.label}
                </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-[10px]">
                <div>
                    <span className="text-muted-foreground">Identity Risk: </span>
                    <span className={entity.sensitivity.identity_risk > 0.7 ? "text-red-400" : "text-emerald-400"}>
                        {(entity.sensitivity.identity_risk * 100).toFixed(0)}%
                    </span>
                </div>
                <div>
                    <span className="text-muted-foreground">Query Need: </span>
                    <span className={entity.sensitivity.query_necessity > 0.6 ? "text-blue-400" : "text-muted-foreground"}>
                        {(entity.sensitivity.query_necessity * 100).toFixed(0)}%
                    </span>
                </div>
                <div>
                    <span className="text-muted-foreground">Re-ID Risk: </span>
                    <span className={entity.sensitivity.reidentification_risk > 0.5 ? "text-amber-400" : "text-emerald-400"}>
                        {(entity.sensitivity.reidentification_risk * 100).toFixed(0)}%
                    </span>
                </div>
            </div>
            {transformation && transformation.transformed !== transformation.original && (
                <div className="mt-1 pt-1 border-t border-border/50 text-[10px]">
                    <span className="text-muted-foreground">Transform: </span>
                    <span className="text-destructive line-through">{transformation.original}</span>
                    <span className="text-muted-foreground"> → </span>
                    <span className={config.text}>{transformation.transformed}</span>
                </div>
            )}
        </div>
    );
}

function RedactedText({ text, transformations }: { 
    text: string;
    transformations?: TransformationInfo[];
}) {
    if (!transformations || transformations.length === 0) return <span>{text}</span>;
    
    // Build map of transformed -> config
    const transformMap = new Map<string, { original: string; strategy: string }>();
    transformations.forEach(t => {
        if (t.transformed !== t.original) {
            transformMap.set(t.transformed, { original: t.original, strategy: t.strategy });
        }
    });
    
    // Also track preserved items
    const preservedSet = new Set<string>();
    transformations.forEach(t => {
        if (t.strategy === "C") {
            preservedSet.add(t.original);
        }
    });
    
    const allKeys = [...transformMap.keys(), ...preservedSet];
    if (allKeys.length === 0) return <span>{text}</span>;
    
    // Sort by length for proper matching
    allKeys.sort((a, b) => b.length - a.length);
    
    const pattern = new RegExp(`(${allKeys.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'g');
    const parts = text.split(pattern);

    const strategyClasses: Record<string, { bg: string; text: string; border: string }> = {
        A: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/30" },
        B: { bg: "bg-amber-500/20", text: "text-amber-400", border: "border-amber-500/30" },
        C: { bg: "bg-blue-500/20", text: "text-blue-400", border: "border-blue-500/30" },
        D: { bg: "bg-purple-500/20", text: "text-purple-400", border: "border-purple-500/30" }
    };

    return (
        <span>
            {parts.map((part, i) => {
                const info = transformMap.get(part);
                if (info) {
                    const classes = strategyClasses[info.strategy] || strategyClasses.A;
                    return (
                        <span 
                            key={i} 
                            className={`${classes.bg} ${classes.text} px-1 rounded mx-0.5 border ${classes.border} font-medium cursor-help`} 
                            title={`Original: ${info.original} | Strategy: ${info.strategy}`}
                        >
                            {part}
                        </span>
                    );
                }
                if (preservedSet.has(part)) {
                    return (
                        <span 
                            key={i} 
                            className="bg-blue-500/20 text-blue-400 px-1 rounded mx-0.5 border border-blue-500/30 font-medium cursor-help"
                            title={`Preserved (Query-Critical)`}
                        >
                            {part}
                        </span>
                    );
                }
                return part;
            })}
        </span>
    );
}
