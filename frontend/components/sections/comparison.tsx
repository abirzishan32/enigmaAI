"use client"

import { Check, X } from "lucide-react"

export function ComparisonSection() {
  const comparisons = [
    { feature: "Data Exposure", traditional: false, EnigmaAI: true, traditionalText: "Plaintext", privacyAsstText: "Encrypted" },
    { feature: "Privacy Guarantee", traditional: false, EnigmaAI: true, traditionalText: "None", privacyAsstText: "Cryptographic" },
    { feature: "Re-identification Risk", traditional: false, EnigmaAI: true, traditionalText: "High", privacyAsstText: "< 10%" },
    { feature: "GDPR Compliance", traditional: false, Enigmaai: true, traditionalText: "Partial", privacyAsstText: "Full" },
    { feature: "Open Source", traditional: false, Enigmaai: true, traditionalText: "Proprietary", privacyAsstText: "MIT License" }
  ]

  return (
    <section className="relative py-24 bg-gradient-to-br from-[#0F1419] to-[#1A1F2E] overflow-hidden">
      <div className="relative z-10 max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            EnigmaAI vs. Traditional AI Systems
          </h2>
          <p className="text-xl text-[#B8C5D6]">
            See how we stack up against conventional approaches
          </p>
        </div>

        {/* Comparison Table */}
        <div className="overflow-x-auto">
          <div className="min-w-[600px] rounded-2xl border border-white/10 overflow-hidden backdrop-blur-sm">
            {/* Table Header */}
            <div className="grid grid-cols-3 bg-white/5 border-b border-white/10">
              <div className="p-6 border-r border-white/10">
                <span className="text-lg font-bold text-white">Feature</span>
              </div>
              <div className="p-6 border-r border-white/10 text-center">
                <span className="text-lg font-bold text-red-400">Traditional AI</span>
              </div>
              <div className="p-6 text-center">
                <span className="text-lg font-bold bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] bg-clip-text text-transparent">
                  EnigmaAI
                </span>
              </div>
            </div>

            {/* Table Rows */}
            {comparisons.map((item, index) => (
              <div 
                key={index}
                className={`grid grid-cols-3 border-b border-white/10 last:border-b-0 hover:bg-white/5 transition-colors ${
                  index % 2 === 0 ? 'bg-white/0' : 'bg-white/[0.02]'
                }`}
              >
                <div className="p-6 border-r border-white/10">
                  <span className="font-semibold text-white">{item.feature}</span>
                </div>
                
                <div className="p-6 border-r border-white/10">
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center">
                      <X className="w-4 h-4 text-red-400" />
                    </div>
                    <span className="text-[#B8C5D6]">{item.traditionalText}</span>
                  </div>
                </div>
                
                <div className="p-6">
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-[#00FFB3]/20 flex items-center justify-center">
                      <Check className="w-4 h-4 text-[#00FFB3]" />
                    </div>
                    <span className="text-[#B8C5D6]">{item.privacyAsstText}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Summary Box */}
        <div className="mt-12 p-8 rounded-2xl bg-gradient-to-br from-[#00D9FF]/10 to-[#00FFB3]/10 border border-[#00D9FF]/30">
          <p className="text-center text-lg text-white">
            <span className="font-bold text-[#00D9FF]">The Result:</span> PrivacyAsst provides comprehensive privacy protection without sacrificing usability, making it the clear choice for privacy-conscious AI applications.
          </p>
        </div>
      </div>
    </section>
  )
}
