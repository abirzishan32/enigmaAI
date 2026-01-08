"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowLeft, BookOpen, Lock, RefreshCw, Cpu, Layers } from "lucide-react"

// --- Data ---
const sections = [
  {
    id: "fhe-basics",
    title: "Fully Homomorphic Encryption",
    icon: Lock,
    content: (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold tracking-tight text-foreground">What is FHE?</h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
                Fully Homomorphic Encryption (FHE) is often called the "Holy Grail" of cryptography. Unlike traditional encryption (encryption-at-rest or in-transit), FHE allows computation to be performed <em>directly on encrypted data</em> without ever needing to decrypt it first.
            </p>
            <div className="bg-card border p-6 rounded-xl">
                <h4 className="font-semibold mb-2">The Analogy</h4>
                <p className="text-sm text-muted-foreground">
                    Imagine working on hazardous materials inside a sealed glovebox. You can manipulate the materials (the data) using the gloves (the operations), but you never actually touch the materials directly, and they never leave the safe environment.
                </p>
            </div>
            <ul className="space-y-3 pt-2">
                <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2" />
                    <span>Server sees only random noise.</span>
                </li>
                <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2" />
                    <span>Data owner holds the secret key.</span>
                </li>
                <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2" />
                    <span>Result of computation is identical to computing on plaintext.</span>
                </li>
            </ul>
        </div>
    )
  },
  {
    id: "shuffling",
    title: "Shuffling Based Algorithm",
    icon: RefreshCw,
    content: (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold tracking-tight text-foreground">Shuffling & Obfuscation</h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
                Beyond standard FHE, our system employs a <strong>Shuffling Based Algorithm</strong> to prevent traffic analysis and metadata leakage. Even if an adversary observes the encrypted data flow, the order and structure of inputs are randomized.
            </p>
             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-card border p-5 rounded-xl">
                    <h4 className="font-medium text-foreground mb-1">Input Shuffling</h4>
                    <p className="text-xs text-muted-foreground">The vector of inputs is randomly permuted before encryption, dissociating the data from its source index.</p>
                </div>
                <div className="bg-card border p-5 rounded-xl">
                    <h4 className="font-medium text-foreground mb-1">Batch Processing</h4>
                    <p className="text-xs text-muted-foreground">Multiple queries are batched together into a single ciphertext, making it impossible to distinguish individual requests.</p>
                </div>
            </div>
            <p className="text-muted-foreground">
                This adds a second layer of defense. Not only is the data unreadable (Ciphertext), but the access patterns themselves are masked, protecting against side-channel attacks.
            </p>
        </div>
    )
  },
  {
    id: "demo-tutorial",
    title: "Demo Tutorial: Digit Recognition",
    icon: Layers,
    content: (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold tracking-tight text-foreground mb-4">How the Demo Works</h2>
                <p className="text-lg text-muted-foreground">
                    Our digit recognition feature demonstrates this technology in a verifiable loop.
                </p>
            </div>

            <div className="space-y-8">
                <div className="relative pl-8 border-l-2 border-slate-200 dark:border-slate-800">
                    <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-indigo-500 ring-4 ring-background" />
                    <h3 className="text-xl font-semibold mb-2">Step 1: Encryption</h3>
                    <p className="text-muted-foreground">
                        When you draw a digit, your browser captures the pixel data. It then uses the <strong>CKKS</strong> Scheme to encrypt these pixels into a polynomial. To the naked eye, this looks like random visual noise.
                    </p>
                </div>
                 <div className="relative pl-8 border-l-2 border-slate-200 dark:border-slate-800">
                    <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-purple-500 ring-4 ring-background" />
                    <h3 className="text-xl font-semibold mb-2">Step 2: Blind Inference</h3>
                    <p className="text-muted-foreground">
                        The encrypted vector is sent to the python backend. The server runs the neural network (weights and biases) <em>homomorphically</em>. It adds and multiplies encrypted numbers.
                    </p>
                    <div className="mt-3 p-3 bg-muted rounded font-mono text-xs overflow-x-auto">
                        Ciphertext(Input) * Plaintext(Weights) + Plaintext(Bias) = Ciphertext(Result)
                    </div>
                </div>
                 <div className="relative pl-8 border-l-2 border-transparent">
                    <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-green-500 ring-4 ring-background" />
                    <h3 className="text-xl font-semibold mb-2">Step 3: Decryption</h3>
                    <p className="text-muted-foreground">
                        The result, still encrypted, is returned to your browser. Only your client possesses the private key to unlock the answer and reveal the prediction.
                    </p>
                </div>
            </div>
             <div className="pt-4">
                <Link href="/digit-recog">
                    <Button className="w-full md:w-auto">Try the Live Demo</Button>
                </Link>
            </div>
        </div>
    )
  }
]

export default function DocumentationPage() {
  const [activeSection, setActiveSection] = useState("fhe-basics")

  return (
    <main className="min-h-screen bg-background pt-24 pb-16">
      <div className="container-custom">
        <div className="flex flex-col md:flex-row items-start gap-12">
            
            {/* LEFT PANEL: Navigation (User Interaction) */}
            <div className="w-full md:w-1/3 lg:w-1/4 sticky top-24 space-y-8">
                <div>
                     <Link href="/" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-6 transition-colors">
                        <ArrowLeft className="w-4 h-4 mr-2" /> Back to Home
                    </Link>
                    <h1 className="text-2xl font-bold mb-2">Documentation</h1>
                    <p className="text-sm text-muted-foreground">Understanding the core technologies behind EnigmaAI.</p>
                </div>

                <nav className="space-y-2">
                    {sections.map((section) => {
                        const Icon = section.icon
                        const isActive = activeSection === section.id
                        return (
                            <button
                                key={section.id}
                                onClick={() => setActiveSection(section.id)}
                                className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all duration-200 ${
                                    isActive 
                                        ? "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 font-medium" 
                                        : "hover:bg-muted text-muted-foreground hover:text-foreground"
                                }`}
                            >
                                <Icon className={`w-5 h-5 ${isActive ? "text-indigo-500" : ""}`} />
                                {section.title}
                            </button>
                        )
                    })}
                </nav>

                <div className="bg-slate-50 dark:bg-slate-900 p-4 rounded-xl border border-border/50">
                    <h4 className="font-semibold text-sm mb-2">Need Help?</h4>
                    <p className="text-xs text-muted-foreground mb-3">Check out our GitHub repository for implementation details.</p>
                    <Button variant="outline" size="sm" className="w-full">View on GitHub</Button>
                </div>
            </div>

            {/* RIGHT PANEL: Content (Technical Transparency) */}
            <div className="w-full md:w-2/3 lg:w-3/4 min-h-[60vh]">
                {sections.map((section) => (
                    activeSection === section.id && (
                        <motion.div
                            key={section.id}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3 }}
                            className="max-w-3xl"
                        >
                            <div className="flex items-center gap-3 mb-6 lg:mb-0 lg:hidden">
                                <section.icon className="w-6 h-6 text-indigo-500" />
                                <h2 className="text-xl font-bold">{section.title}</h2>
                            </div>
                           {section.content}
                        </motion.div>
                    )
                ))}
            </div>

        </div>
      </div>
    </main>
  )
}
