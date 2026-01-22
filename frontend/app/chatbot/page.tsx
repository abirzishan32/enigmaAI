// Context-Aware Redaction
"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp, Shield, Lock, Eye, EyeOff, Bot } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  metadata?: {
    original?: string;
    redacted?: string;
    pii_map?: Record<string, string>;
    preserved_items?: Record<string, string>;
    raw_response?: string;
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
      
      if (data.error) throw new Error(data.error);

      // Add Assistant Response
      const botMsg: ChatMessage = {
        role: "assistant",
        content: data.llm_response_restored, // Show restored by default
        metadata: {
          original: data.original_prompt,
          redacted: data.redacted_prompt,
          pii_map: data.pii_map,
          preserved_items: data.preserved_items,
          raw_response: data.llm_response_raw
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
      {/* Header */}
      <header className="border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50 shrink-0">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary/10 p-2 rounded-lg border border-primary/20">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-bold text-lg">Secure Chat</h1>
              <div className="text-xs text-muted-foreground flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Privacy Layer Active
              </div>
            </div>
          </div>
          <div className="text-xs text-primary bg-primary/10 px-3 py-1 rounded-full border border-primary/20 hidden sm:block">
            PII Redaction Engine: Spacy NER
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 w-full max-w-5xl mx-auto p-4 flex flex-col gap-4 overflow-hidden">
        

        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center text-center space-y-4 opacity-50">
            <Shield className="w-16 h-16 text-muted-foreground" />
            <p className="text-muted-foreground max-w-sm">
              Your messages are sanitized locally before being sent to the AI. Personal details like names and locations are masked.
            </p>
          </div>
        )}

        <div className="flex-1 overflow-y-auto space-y-6 pr-2" ref={scrollRef}>
          {messages.map((msg, idx) => (
            <MessageItem key={idx} msg={msg} />
          ))}
          {isLoading && (
             <div className="flex gap-4">
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

        {/* Input Area */}
        <div className="bg-card border border-input rounded-2xl p-2 relative shadow-2xl shrink-0">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
                if(e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                }
            }}
            placeholder="Ask anything... (Try: 'My name is Alice and I live in Paris')"
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
        
      </main>
    </div>
  );
}

function MessageItem({ msg }: { msg: ChatMessage }) {
    const isUser = msg.role === "user";
    const [showDebug, setShowDebug] = useState(false);
    
    return (
        <div className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center border shrink-0 ${
                isUser 
                ? "bg-secondary border-secondary" 
                : "bg-primary/10 border-primary/20"
            }`}>
               {isUser ? <div className="w-4 h-4 bg-secondary-foreground/50 rounded-full" /> : <Bot className="w-4 h-4 text-primary" />}
            </div>

            <div className={`flex flex-col gap-2 max-w-[85%] ${isUser ? "items-end" : "items-start w-full"}`}>
                <div className={`text-sm max-w-none ${
                    isUser 
                    ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-none px-3 py-2.5 leading-relaxed [&>p]:mb-2 [&>p:last-child]:mb-0 [&>ul]:list-disc [&>ul]:pl-4" 
                    : "text-foreground px-0 py-2 leading-8 w-full prose dark:prose-invert"
                    }`}>
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>

                {/* Privacy Inspector (Assistant Only) */}
                {!isUser && msg.metadata && (msg.metadata.original !== msg.metadata.redacted) && (
                    <div className="w-full">
                        <button 
                             onClick={() => setShowDebug(!showDebug)}
                             className="text-xs flex items-center gap-1.5 text-primary hover:text-primary/80 transition-colors bg-primary/5 px-3 py-1.5 rounded-lg border border-primary/10"
                        >
                            {showDebug ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                            {showDebug ? "Hide Privacy Trace" : "View Sensitive Data Handling"}
                        </button>
                        
                        <AnimatePresence>
                            {showDebug && (
                                <motion.div 
                                    initial={{ height: 0, opacity: 0 }} 
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="overflow-hidden mt-2"
                                >
                                    <div className="bg-card rounded-xl border border-primary/20 p-4 text-xs font-mono space-y-4 shadow-inner">
                                        
                                        {/* What User Sent */}
                                        <div className="space-y-1">
                                            <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold">Original Input</div>
                                            <div className="p-2 bg-destructive/10 border border-destructive/20 rounded text-destructive break-words">
                                                {msg.metadata.original}
                                            </div>
                                        </div>

                                        {/* Arrow */}
                                        <div className="flex justify-center text-muted-foreground">
                                            <div className="bg-muted p-1 rounded-full border border-border">
                                                <Lock className="w-3 h-3" />
                                            </div>
                                        </div>

                                        {/* What Gemini Saw */}
                                        <div className="space-y-1">
                                            <div className="text-muted-foreground uppercase tracking-widest text-[10px] font-bold flex justify-between">
                                                <span>Sent to Gemini API</span>
                                                <span className="text-emerald-500">Synthetic Data Active</span>
                                            </div>
                                            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded text-emerald-600 dark:text-emerald-400 break-words">
                                                {/* Highlight Redactions and Preservation */}
                                                <RedactedText 
                                                    text={msg.metadata.redacted || ""} 
                                                    piiMap={msg.metadata.pii_map} 
                                                    preservedItems={msg.metadata.preserved_items}
                                                />
                                            </div>
                                        </div>
                                        
                                        {/* Detected Entities */}
                                        {msg.metadata.pii_map && Object.keys(msg.metadata.pii_map).length > 0 && (
                                            <div className="pt-2 border-t border-border">
                                                <div className="text-muted-foreground mb-2">Detected Entities (Stored Locally):</div>
                                                <div className="grid grid-cols-2 gap-2">
                                                    {Object.entries(msg.metadata.pii_map).map(([fakeVal, original]) => (
                                                        <div key={fakeVal} className="flex justify-between bg-muted p-1.5 rounded border border-border">
                                                            <span className="text-primary truncate max-w-[45%]">{original}</span>
                                                            <span className="text-muted-foreground">→</span>
                                                            <span className="text-emerald-500 truncate pl-2 max-w-[45%]">{fakeVal}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Preserved Entities */}
                                        {msg.metadata.preserved_items && Object.keys(msg.metadata.preserved_items).length > 0 && (
                                            <div className="pt-2 border-t border-border">
                                                <div className="text-muted-foreground mb-2 flex items-center gap-2">
                                                    <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                                    Preserved Context (Public):
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    {Object.entries(msg.metadata.preserved_items).map(([text, label]) => (
                                                        <div key={text} className="flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 px-2 py-1 rounded text-blue-400">
                                                            <span>{text}</span>
                                                            <span className="text-[9px] uppercase opacity-50 bg-blue-500/20 px-1 rounded">{label}</span>
                                                        </div>
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

function RedactedText({ text, piiMap, preservedItems }: { 
    text: string, 
    piiMap?: Record<string, string>,
    preservedItems?: Record<string, string>
}) {
    const piiKeys = piiMap ? Object.keys(piiMap) : [];
    const preservedKeys = preservedItems ? Object.keys(preservedItems) : [];
    
    // Combine keys (Fake Tokens AND Preserved Real Text)
    const allKeys = [...piiKeys, ...preservedKeys].sort((a, b) => b.length - a.length);

    if (allKeys.length === 0) return <span>{text}</span>;

    // Escape and build pattern
    const pattern = new RegExp(`(${allKeys.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'g');
    
    // ... splitting logic same ...
    
    const parts = text.split(pattern);

    return (
        <span>
            {parts.map((part, i) => {
                if (piiMap && piiMap.hasOwnProperty(part)) {
                     return (
                        <span key={i} className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-1 rounded mx-0.5 border border-emerald-500/20 font-medium cursor-help" title={`Original: ${piiMap[part]}`}>
                            {part}
                        </span>
                     )
                }
                if (preservedItems && preservedItems.hasOwnProperty(part)) {
                    return (
                       <span key={i} className="bg-blue-500/10 text-blue-400 px-1 rounded mx-0.5 border border-blue-500/20 font-medium cursor-help" title={`Preserved Entity: ${preservedItems[part]} (Query Context)`}>
                           {part}
                       </span>
                    )
               }
                return part;
            })}
        </span>
    )
}
