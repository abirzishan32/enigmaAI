"use client"

import { Eye, Server, UserX } from "lucide-react"

export function ProblemSection() {
  const problems = [
    {
      icon: Eye,
      title: "Data Exposure",
      description: "LLM agents process sensitive personal information in plaintext, risking unauthorized access and privacy breaches"
    },
    {
      icon: Server,
      title: "Third-Party Risks",
      description: "External AI services collect, store, and potentially misuse user data without transparency"
    },
    {
      icon: UserX,
      title: "Identity Leakage",
      description: "Demographic attributes in prompts enable re-identification attacks and profiling"
    }
  ]

  return (
    <section className="relative py-24 bg-[#0A1628] overflow-hidden">
      {/* Grid Pattern Background */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(#00D9FF 1px, transparent 1px), linear-gradient(90deg, #00D9FF 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            The Privacy Crisis in AI
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] mx-auto"></div>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {problems.map((problem, index) => {
            const Icon = problem.icon
            return (
              <div 
                key={index}
                className="group relative p-8 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:border-[#00D9FF]/50 transition-all duration-300 hover:shadow-xl hover:shadow-[#00D9FF]/20"
              >
                {/* Glow Effect on Hover */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-[#00D9FF]/0 to-[#00FFB3]/0 group-hover:from-[#00D9FF]/10 group-hover:to-[#00FFB3]/10 transition-all duration-300"></div>
                
                <div className="relative z-10">
                  <div className="w-16 h-16 mb-6 rounded-xl bg-gradient-to-br from-[#00D9FF]/20 to-[#00FFB3]/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <Icon className="w-8 h-8 text-[#00D9FF]" />
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-4">
                    {problem.title}
                  </h3>
                  
                  <p className="text-[#B8C5D6] leading-relaxed">
                    {problem.description}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
