"use client"

import { Lock, Shuffle, Check } from "lucide-react"

export function SolutionSection() {
  return (
    <section className="py-20 md:py-32" id="solution">
      <div className="container-custom">
        <div className="text-center mb-16">
          <h2 className="font-bold text-foreground mb-4">
            Dual Privacy Mechanisms
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Choose the right protection level for your specific use case
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Homomorphic Encryption */}
          <div className="p-8 rounded-lg border border-border bg-card">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <div className="h-1.5 w-1.5 rounded-full bg-primary" />
              <span className="text-sm font-medium">Cryptographic Privacy</span>
            </div>

            <div className="mb-6">
              <div className="inline-flex p-3 rounded-lg bg-primary/10">
                <Lock className="h-7 w-7 text-primary" strokeWidth={2} />
              </div>
            </div>

            <h3 className="text-2xl font-semibold mb-3">
              Homomorphic Encryption
            </h3>

            <p className="text-muted-foreground mb-6">
              Process text on encrypted data without decryption. Server never sees your plaintext.
            </p>

            <ul className="space-y-3 mb-6">
              {[
                "128-bit post-quantum security",
                "Zero plaintext exposure",
                "CKKS scheme implementation",
                "TenSEAL powered"
              ].map((feature, index) => (
                <li key={index} className="flex items-center gap-3 text-sm">
                  <Check className="h-4 w-4 text-primary flex-shrink-0" strokeWidth={2.5} />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            <div className="p-4 rounded-lg bg-muted border border-border">
              <p className="text-xs font-medium text-muted-foreground mb-1">Best For:</p>
              <p className="text-sm">Medical records, financial data, legal documents</p>
            </div>
          </div>

          {/* Attribute Shuffling */}
          <div className="p-8 rounded-lg border border-border bg-card">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <div className="h-1.5 w-1.5 rounded-full bg-primary" />
              <span className="text-sm font-medium">Statistical Privacy</span>
            </div>

            <div className="mb-6">
              <div className="inline-flex p-3 rounded-lg bg-primary/10">
                <Shuffle className="h-7 w-7 text-primary" strokeWidth={2} />
              </div>
            </div>

            <h3 className="text-2xl font-semibold mb-3">
              Attribute Shuffling
            </h3>

            <p className="text-muted-foreground mb-6">
              Randomize demographic attributes while preserving semantic meaning via t-closeness.
            </p>

            <ul className="space-y-3 mb-6">
              {[
                "t-closeness ≤ 0.3 guarantee",
                "12 attribute categories",
                "Re-identification risk < 10%",
                "Semantic coherence maintained"
              ].map((feature, index) => (
                <li key={index} className="flex items-center gap-3 text-sm">
                  <Check className="h-4 w-4 text-primary flex-shrink-0" strokeWidth={2.5} />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            <div className="p-4 rounded-lg bg-muted border border-border">
              <p className="text-xs font-medium text-muted-foreground mb-1">Best For:</p>
              <p className="text-sm">Image generation, chatbots, general AI tasks</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
