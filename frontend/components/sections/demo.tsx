"use client"

import { AlertTriangle, CheckCircle, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export function DemoSection() {
  return (
    <section className="relative py-24 bg-[#0A1628] overflow-hidden">
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Try It Yourself
          </h2>
          <p className="text-xl text-[#B8C5D6]">
            See the difference privacy protection makes
          </p>
        </div>

        {/* Split Screen Comparison */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* WITHOUT PRIVACY */}
          <div className="relative p-8 rounded-3xl bg-gradient-to-br from-red-500/10 to-red-500/5 border border-red-500/30">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
              <AlertTriangle className="w-6 h-6 text-red-400" />
              <h3 className="text-2xl font-bold text-white">Without Privacy</h3>
            </div>

            {/* Input Box */}
            <div className="mb-6 p-4 rounded-xl bg-black/30 border border-white/10">
              <p className="text-sm text-[#B8C5D6] mb-2 font-semibold">User Input:</p>
              <p className="text-white">
                "Generate image of 35-year-old Asian male doctor"
              </p>
            </div>

            {/* Attribute Display */}
            <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30">
              <p className="text-sm text-red-400 mb-3 font-semibold">Exposed Attributes:</p>
              <div className="flex flex-wrap gap-2">
                {["Age: 35", "Race: Asian", "Gender: Male", "Occupation: Doctor"].map((attr, i) => (
                  <span key={i} className="px-3 py-1 rounded-full bg-red-500/20 text-red-300 text-sm border border-red-500/30">
                    {attr}
                  </span>
                ))}
              </div>
            </div>

            {/* Output Placeholder */}
            <div className="mb-6 aspect-video rounded-xl bg-gradient-to-br from-gray-800 to-gray-900 border border-white/10 flex items-center justify-center">
              <span className="text-[#B8C5D6]">Generated Image</span>
            </div>

            {/* Warning */}
            <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-400 font-semibold mb-1">Privacy Risk Detected</p>
                <p className="text-sm text-red-300/80">Your personal attributes are exposed and can be used for profiling!</p>
              </div>
            </div>
          </div>

          {/* WITH PRIVACY */}
          <div className="relative p-8 rounded-3xl bg-gradient-to-br from-[#00FFB3]/10 to-[#00D9FF]/10 border border-[#00FFB3]/30">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
              <CheckCircle className="w-6 h-6 text-[#00FFB3]" />
              <h3 className="text-2xl font-bold text-white">With EnigmaAI</h3>
            </div>

            {/* Input Box */}
            <div className="mb-6 p-4 rounded-xl bg-black/30 border border-white/10">
              <p className="text-sm text-[#B8C5D6] mb-2 font-semibold">User Input:</p>
              <p className="text-white">
                "Generate image of 35-year-old Asian male doctor"
              </p>
            </div>

            {/* Shuffling Animation */}
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex-1 h-0.5 bg-gradient-to-r from-[#00D9FF] to-[#00FFB3]"></div>
                <span className="text-sm text-[#00D9FF] font-semibold">Attribute Shuffling</span>
                <div className="flex-1 h-0.5 bg-gradient-to-r from-[#00FFB3] to-[#00D9FF]"></div>
              </div>
              
              <div className="p-4 rounded-xl bg-[#00FFB3]/10 border border-[#00FFB3]/30">
                <p className="text-sm text-[#00FFB3] mb-3 font-semibold">Protected Attributes:</p>
                <div className="flex flex-wrap gap-2">
                  {["Age: 42", "Race: Hispanic", "Gender: Male", "Occupation: Teacher"].map((attr, i) => (
                    <span key={i} className="px-3 py-1 rounded-full bg-[#00FFB3]/20 text-[#00FFB3] text-sm border border-[#00FFB3]/30 animate-fade-in" style={{ animationDelay: `${i * 100}ms` }}>
                      {attr}
                    </span>
                  ))}
                </div>
                <p className="text-xs text-[#B8C5D6] mt-3 italic">t-closeness: 0.28 ≤ 0.3 ✓</p>
              </div>
            </div>

            {/* Output Placeholder */}
            <div className="mb-6 aspect-video rounded-xl bg-gradient-to-br from-gray-800 to-gray-900 border border-[#00FFB3]/30 flex items-center justify-center">
              <span className="text-[#B8C5D6]">Generated Image (Randomized)</span>
            </div>

            {/* Success */}
            <div className="flex items-start gap-3 p-4 rounded-xl bg-[#00FFB3]/10 border border-[#00FFB3]/30">
              <CheckCircle className="w-5 h-5 text-[#00FFB3] flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-[#00FFB3] font-semibold mb-1">Privacy Preserved!</p>
                <p className="text-sm text-[#B8C5D6]">Your true attributes are protected with statistical guarantees.</p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Button */}
        <div className="text-center">
          <Button 
            size="lg"
            className="bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] hover:from-[#00D9FF]/90 hover:to-[#00FFB3]/90 text-black font-bold text-lg px-8 py-6 shadow-lg shadow-[#00D9FF]/50 hover:shadow-xl hover:shadow-[#00D9FF]/60 transition-all"
          >
            Launch Interactive Demo
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </div>
      </div>
    </section>
  )
}
