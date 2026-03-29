"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowUp, Bot, Shield, Lock, Eye, Zap, BarChart3,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

type AnonymizationAction = "Retain" | "Replace" | "Encrypt" | "Delete";

interface WordToken {
  word:   string;
  index:  number;
  plrs:   number;   // Privacy Leakage Risk Score   0–1
  ciis:   number;   // Contextual Information Importance Score 0–1
  trs:    number;   // Task Relevance Score          0–1
  action: AnonymizationAction;
}

interface MessageMeta {
  original_prompt:  string;
  sanitized_prompt: string;
  word_tokens:      WordToken[];
}

interface ChatMessage {
  role:     "user" | "assistant";
  content:  string;
  meta?:    MessageMeta;
}

// ─────────────────────────────────────────────────────────────
// Action colour palette
// ─────────────────────────────────────────────────────────────

const ACTION_STYLES: Record<
  AnonymizationAction,
  { chip: string; pill: string; dot: string; label: string }
> = {
  Retain:  {
    chip:  "bg-emerald-500/15 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/25",
    pill:  "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    dot:   "bg-emerald-400",
    label: "Retain",
  },
  Replace: {
    chip:  "bg-amber-500/15 text-amber-300 border-amber-500/30 hover:bg-amber-500/25",
    pill:  "bg-amber-500/20 text-amber-400 border-amber-500/30",
    dot:   "bg-amber-400",
    label: "Replace",
  },
  Encrypt: {
    chip:  "bg-purple-500/15 text-purple-300 border-purple-500/30 hover:bg-purple-500/25",
    pill:  "bg-purple-500/20 text-purple-400 border-purple-500/30",
    dot:   "bg-purple-400",
    label: "Encrypt",
  },
  Delete:  {
    chip:  "bg-red-500/15 text-red-300 border-red-500/30 hover:bg-red-500/25 line-through",
    pill:  "bg-red-500/20 text-red-400 border-red-500/30",
    dot:   "bg-red-400",
    label: "Delete",
  },
};

// ─────────────────────────────────────────────────────────────
// Page component
// ─────────────────────────────────────────────────────────────

export default function ALSAChatPage() {
  const [messages,  setMessages ] = useState<ChatMessage[]>([]);
  const [input,     setInput    ] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res  = await fetch("http://localhost:8000/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message: userMsg.content }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");

      const botMsg: ChatMessage = {
        role:    "assistant",
        content: data.llm_response,
        meta: {
          original_prompt:  data.original_prompt,
          sanitized_prompt: data.sanitized_prompt,
          word_tokens:      data.word_tokens,
        },
      };
      setMessages(prev => [...prev, botMsg]);

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: `⚠️ Error: ${message}`,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-[calc(100vh-65px)] flex flex-col overflow-hidden bg-background text-foreground">

      {/* ── Scrollable message area ───────────── */}
      <div className="flex-1 overflow-y-auto" ref={scrollRef}>
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">

          {messages.length === 0 && <EmptyState />}

          {messages.map((msg, idx) => (
            <MessageItem key={idx} msg={msg} />
          ))}

          {isLoading && <TypingIndicator />}
        </div>
      </div>

      {/* ── Input bar ────────────────────────── */}
      <div className="border-t border-border bg-background/80 backdrop-blur px-4 py-3">
        <div className="max-w-4xl mx-auto">
          <div className="bg-card border border-input rounded-2xl p-2 shadow-xl relative">
            <Textarea
              id="alsa-prompt-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }}
              placeholder="Type a message… every word will be individually evaluated by ALSA."
              className="bg-transparent border-none focus-visible:ring-0 min-h-[52px] max-h-[160px] resize-none pr-14 text-sm text-card-foreground placeholder:text-muted-foreground"
            />
            <Button
              id="alsa-send-btn"
              size="icon"
              onClick={handleSubmit}
              disabled={isLoading || !input.trim()}
              className="absolute right-2 bottom-2 rounded-xl w-10 h-10 transition-all hover:scale-105"
            >
              <ArrowUp className="w-5 h-5" />
            </Button>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-3 mt-2">
            {(Object.entries(ACTION_STYLES) as [AnonymizationAction, typeof ACTION_STYLES[AnonymizationAction]][]).map(([action, s]) => (
              <span key={action} className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border ${s.pill}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
                {action}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// EmptyState
// ─────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-5 opacity-60">
      <div className="p-4 bg-primary/10 rounded-2xl border border-primary/20">
        <Shield className="w-12 h-12 text-primary" />
      </div>
      <div className="space-y-2">
        <h2 className="text-lg font-semibold">ALSA Privacy Inspector</h2>
        <p className="text-sm text-muted-foreground max-w-xs">
          Every word in your prompt is individually evaluated for Privacy Leakage Risk,
          Contextual Importance, and Task Relevance before reaching the LLM.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {(Object.entries(ACTION_STYLES) as [AnonymizationAction, typeof ACTION_STYLES[AnonymizationAction]][]).map(([action, s]) => (
          <span key={action} className={`text-xs px-3 py-1 rounded-full border ${s.pill}`}>
            {action}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TypingIndicator
// ─────────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <BotAvatar />
      <div className="bg-card border border-border rounded-2xl rounded-tl-none px-4 py-3">
        <div className="flex gap-1 items-center">
          {[0, 150, 300].map(delay => (
            <span
              key={delay}
              className="w-2 h-2 bg-primary rounded-full animate-bounce"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// BotAvatar / UserAvatar
// ─────────────────────────────────────────────────────────────

function BotAvatar() {
  return (
    <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
      <Bot className="w-4 h-4 text-primary" />
    </div>
  );
}

function UserAvatar() {
  return (
    <div className="w-8 h-8 rounded-full bg-secondary border border-secondary-foreground/10 flex items-center justify-center shrink-0">
      <div className="w-4 h-4 bg-secondary-foreground/40 rounded-full" />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// MessageItem
// ─────────────────────────────────────────────────────────────

function MessageItem({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {isUser ? <UserAvatar /> : <BotAvatar />}

      <div className={`flex flex-col gap-3 max-w-[92%] ${isUser ? "items-end" : "items-start w-full"}`}>

        {/* Bubble */}
        <div className={
          isUser
            ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-none px-4 py-2.5 text-sm leading-relaxed"
            : "text-foreground text-sm leading-8 w-full prose dark:prose-invert max-w-none"
        }>
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>

        {/* Word Inspector (assistant only) */}
        {!isUser && msg.meta && (
          <WordInspectorPanel meta={msg.meta} />
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// WordInspectorPanel  ← The Core Visualizer
// ─────────────────────────────────────────────────────────────

function WordInspectorPanel({ meta }: { meta: MessageMeta }) {
  const [selectedWord, setSelectedWord] = useState<WordToken | null>(null);
  const [showSanitized, setShowSanitized] = useState(false);

  const { word_tokens: tokens, original_prompt, sanitized_prompt } = meta;

  // Action counts
  const counts = tokens.reduce(
    (acc, t) => { acc[t.action] = (acc[t.action] ?? 0) + 1; return acc; },
    {} as Record<AnonymizationAction, number>,
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="w-full bg-card border border-border rounded-2xl overflow-hidden shadow-lg"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2">
          <div className="p-1 bg-primary/10 rounded-md">
            <Zap className="w-3.5 h-3.5 text-primary" />
          </div>
          <span className="text-xs font-semibold">ALSA Word Inspector</span>
          <span className="text-[10px] text-muted-foreground">({tokens.length} tokens)</span>
        </div>

        {/* Action count pills */}
        <div className="flex gap-1.5">
          {(Object.entries(counts) as [AnonymizationAction, number][]).map(([action, n]) => (
            <span key={action} className={`text-[10px] px-2 py-0.5 rounded-full border ${ACTION_STYLES[action].pill}`}>
              {n} {action}
            </span>
          ))}
        </div>
      </div>

      {/* Word Chip Grid */}
      <div className="px-4 py-3 flex flex-wrap gap-2">
        {tokens.map(token => (
          <WordChip
            key={token.index}
            token={token}
            isSelected={selectedWord?.index === token.index}
            onClick={() => setSelectedWord(prev => prev?.index === token.index ? null : token)}
          />
        ))}
      </div>

      {/* Score Detail Panel (expands when a word is selected) */}
      <AnimatePresence>
        {selectedWord && (
          <motion.div
            key={selectedWord.index}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-border"
          >
            <WordScoreDetail token={selectedWord} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prompt Comparison (toggle) */}
      <div className="border-t border-border">
        <button
          onClick={() => setShowSanitized(v => !v)}
          className="w-full flex items-center justify-between px-4 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Eye className="w-3 h-3" />
            <span>Prompt Comparison</span>
          </div>
          <span className="text-[10px] opacity-60">{showSanitized ? "hide" : "show"}</span>
        </button>

        <AnimatePresence>
          {showSanitized && (
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: "auto" }}
              exit={{ height: 0 }}
              className="overflow-hidden"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 px-4 pb-4">
                <PromptBlock label="Original (Sensitive)" color="red" text={original_prompt} />
                <PromptBlock label="Sent to LLM (Sanitized)" color="emerald" text={sanitized_prompt} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────
// WordChip
// ─────────────────────────────────────────────────────────────

function WordChip({ token, isSelected, onClick }: {
  token:      WordToken;
  isSelected: boolean;
  onClick:    () => void;
}) {
  const s = ACTION_STYLES[token.action];

  return (
    <button
      id={`word-chip-${token.index}`}
      onClick={onClick}
      title={`${token.action} | PLRS: ${token.plrs.toFixed(2)} | CIIS: ${token.ciis.toFixed(2)} | TRS: ${token.trs.toFixed(2)}`}
      className={`
        relative inline-flex items-center gap-1.5 px-2.5 py-1
        rounded-lg border text-xs font-medium transition-all cursor-pointer
        ${s.chip}
        ${isSelected ? "ring-2 ring-primary ring-offset-1 ring-offset-background scale-105" : ""}
      `}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot} shrink-0`} />
      {token.word}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────
// WordScoreDetail
// ─────────────────────────────────────────────────────────────

function WordScoreDetail({ token }: { token: WordToken }) {
  const s = ACTION_STYLES[token.action];

  const scores: { label: string; abbr: string; value: number; desc: string; color: string }[] = [
    {
      label: "Privacy Leakage Risk",
      abbr:  "PLRS",
      value: token.plrs,
      desc:  "Likelihood this word leaks personal or sensitive information",
      color: "red",
    },
    {
      label: "Contextual Importance",
      abbr:  "CIIS",
      value: token.ciis,
      desc:  "How important this word is for preserving the prompt's semantic context",
      color: "blue",
    },
    {
      label: "Task Relevance",
      abbr:  "TRS",
      value: token.trs,
      desc:  "How critical this word is for the LLM to perform the intended task",
      color: "purple",
    },
  ];

  const barColor: Record<string, string> = {
    red:    "bg-red-500",
    blue:   "bg-blue-500",
    purple: "bg-purple-500",
  };

  return (
    <div className="px-4 py-3 bg-muted/20 space-y-3">
      {/* Word header */}
      <div className="flex items-center gap-3">
        <div className={`px-3 py-1.5 rounded-lg border text-sm font-bold ${s.pill}`}>
          "{token.word}"
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`text-xs font-semibold ${s.pill.includes("emerald") ? "text-emerald-400" : s.pill.includes("amber") ? "text-amber-400" : s.pill.includes("purple") ? "text-purple-400" : "text-red-400"}`}>
            Action:
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full border ${s.pill}`}>
            {token.action}
          </span>
        </div>
        <span className="text-[10px] text-muted-foreground ml-auto">Token #{token.index}</span>
      </div>

      {/* Score bars */}
      <div className="space-y-2">
        {scores.map(score => (
          <div key={score.abbr} className="space-y-0.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
                  {score.abbr}
                </span>
                <span className="text-[10px] text-muted-foreground hidden sm:inline">
                  — {score.label}
                </span>
              </div>
              <span className="text-xs font-bold tabular-nums">
                {score.value.toFixed(4)}
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${barColor[score.color]}`}
                initial={{ width: 0 }}
                animate={{ width: `${score.value * 100}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>
            <p className="text-[9px] text-muted-foreground/70">{score.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// PromptBlock
// ─────────────────────────────────────────────────────────────

function PromptBlock({ label, color, text }: { label: string; color: string; text: string }) {
  const colorMap: Record<string, { border: string; label: string; text: string; bg: string; icon: string }> = {
    red:     { border: "border-red-500/30",     label: "text-red-400",     text: "text-red-300/80",     bg: "bg-red-500/5",     icon: "🔓" },
    emerald: { border: "border-emerald-500/30", label: "text-emerald-400", text: "text-emerald-300/80", bg: "bg-emerald-500/5", icon: "🔒" },
  };
  const c = colorMap[color] || colorMap.red;

  return (
    <div className={`rounded-lg border ${c.border} ${c.bg} p-3 space-y-1`}>
      <div className={`text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 ${c.label}`}>
        <Lock className="w-2.5 h-2.5" />
        {label}
      </div>
      <p className={`text-[11px] font-mono break-all leading-relaxed ${c.text}`}>{text}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// (unused in main flow — kept for convenience in Phase 2)
// ─────────────────────────────────────────────────────────────

function ScoreBadge({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] font-mono font-bold">{value.toFixed(2)}</div>
      <div className="text-[9px] text-muted-foreground">{label}</div>
      <BarChart3 className="w-3 h-3 text-muted-foreground/50 mt-0.5" />
    </div>
  );
}
