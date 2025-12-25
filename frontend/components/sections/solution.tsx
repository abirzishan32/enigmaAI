"use client"

import { Lock, Shuffle, Check } from "lucide-react"

export function SolutionSection() {
  return (
    <section className="relative py-24 bg-gradient-to-br from-[#1A1F2E] to-[#0F1419] overflow-hidden">
      {/* Geometric Pattern Background */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 35px, #00FFB3 35px, #00FFB3 36px)`
        }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Dual Privacy Mechanisms
          </h2>
          <p className="text-xl text-[#B8C5D6]">
            Choose the right protection for your use case
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Homomorphic Encryption */}
          <div className="relative p-8 rounded-3xl bg-gradient-to-br from-[#0A1628] to-[#1A1F2E] border border-[#00D9FF]/30 overflow-hidden group hover:border-[#00D9FF] transition-all duration-300">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#00D9FF]/10 border border-[#00D9FF]/30 mb-6">
              <div className="w-2 h-2 bg-[#00D9FF] rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-[#00D9FF]">Cryptographic Privacy</span>
            </div>

            {/* Icon */}
            <div className="w-20 h-20 mb-6 rounded-2xl bg-gradient-to-br from-[#00D9FF]/20 to-[#00FFB3]/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Lock className="w-10 h-10 text-[#00D9FF]" />
            </div>

            <h3 className="text-3xl font-bold text-white mb-4">
              Text-Enabled Homomorphic Encryption
            </h3>

            <p className="text-[#B8C5D6] mb-6 leading-relaxed">
              Process text on encrypted data without decryption. Server never sees your plaintext.
            </p>

            {/* Features List */}
            <ul className="space-y-3 mb-6">
              {[
                "128-bit post-quantum security",
                "Zero plaintext exposure",
                "CKKS scheme implementation",
                "TenSEAL powered"
              ].map((feature, index) => (
                <li key={index} className="flex items-center gap-3 text-[#B8C5D6]">
                  <Check className="w-5 h-5 text-[#00FFB3] flex-shrink-0" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            {/* Use Case */}
            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
              <p className="text-sm font-semibold text-[#00D9FF] mb-1">Best For:</p>
              <p className="text-sm text-[#B8C5D6]">Medical records, financial data, legal documents</p>
            </div>

            {/* Animated Background Glow */}
            <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-[#00D9FF] rounded-full blur-[120px] opacity-20 group-hover:opacity-30 transition-opacity duration-300"></div>
          </div>

          {/* Attribute Shuffling */}
          <div className="relative p-8 rounded-3xl bg-gradient-to-br from-[#0A1628] to-[#1A1F2E] border border-[#00FFB3]/30 overflow-hidden group hover:border-[#00FFB3] transition-all duration-300">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#00FFB3]/10 border border-[#00FFB3]/30 mb-6">
              <div className="w-2 h-2 bg-[#00FFB3] rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-[#00FFB3]">Statistical Privacy</span>
            </div>

            {/* Icon */}
            <div className="w-20 h-20 mb-6 rounded-2xl bg-gradient-to-br from-[#00FFB3]/20 to-[#00D9FF]/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Shuffle className="w-10 h-10 text-[#00FFB3]" />
            </div>

            <h3 className="text-3xl font-bold text-white mb-4">
              Expanded Attribute Shuffling
            </h3>

            <p className="text-[#B8C5D6] mb-6 leading-relaxed">
              Randomize 12 demographic attributes while preserving semantic meaning via t-closeness.
            </p>

            {/* Features List */}
            <ul className="space-y-3 mb-6">
              {[
                "t-closeness ≤ 0.3 guarantee",
                "12 attribute categories",
                "Re-identification risk < 10%",
                "Semantic coherence maintained"
              ].map((feature, index) => (
                <li key={index} className="flex items-center gap-3 text-[#B8C5D6]">
                  <Check className="w-5 h-5 text-[#00FFB3] flex-shrink-0" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            {/* Use Case */}
            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
              <p className="text-sm font-semibold text-[#00FFB3] mb-1">Best For:</p>
              <p className="text-sm text-[#B8C5D6]">Image generation, chatbots, general AI tasks</p>
            </div>

            {/* Animated Background Glow */}
            <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-[#00FFB3] rounded-full blur-[120px] opacity-20 group-hover:opacity-30 transition-opacity duration-300"></div>
          </div>
        </div>
      </div>
    </section>
  )
}
