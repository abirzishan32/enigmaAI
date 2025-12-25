"use client"

import { User, Shield, Brain, LockKeyhole, ArrowRight } from "lucide-react"

export function HowItWorksSection() {
  const steps = [
    {
      icon: User,
      title: "User Input",
      description: "Submit sensitive query",
      color: "#00D9FF"
    },
    {
      icon: Shield,
      title: "Privacy Layer",
      description: "Encrypt or shuffle attributes",
      color: "#00FFB3"
    },
    {
      icon: Brain,
      title: "AI Processing",
      description: "Blind computation",
      color: "#FFD700"
    },
    {
      icon: LockKeyhole,
      title: "Secure Output",
      description: "Decrypt results locally",
      color: "#00D9FF"
    }
  ]

  return (
    <section className="relative py-24 bg-[#0A1628] overflow-hidden">
      {/* Particle Background */}
      <div className="absolute inset-0">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-[#00D9FF] rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${5 + Math.random() * 10}s`,
              opacity: 0.3
            }}
          ></div>
        ))}
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Privacy-Preserving Pipeline
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] mx-auto"></div>
        </div>

        {/* Steps */}
        <div className="relative">
          <div className="grid md:grid-cols-4 gap-8">
            {steps.map((step, index) => {
              const Icon = step.icon
              return (
                <div key={index} className="relative">
                  {/* Connecting Line */}
                  {index < steps.length - 1 && (
                    <div className="hidden md:block absolute top-20 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] opacity-30">
                      <ArrowRight className="absolute right-0 -top-3 w-6 h-6 text-[#00D9FF] animate-pulse" />
                    </div>
                  )}

                  <div className="relative group">
                    {/* Step Card */}
                    <div className="p-6 rounded-2xl bg-gradient-to-br from-white/5 to-white/0 backdrop-blur-sm border border-white/10 hover:border-[#00D9FF]/50 transition-all duration-300 hover:shadow-xl hover:shadow-[#00D9FF]/20">
                      {/* Icon Container */}
                      <div 
                        className="w-20 h-20 mx-auto mb-6 rounded-xl flex items-center justify-center relative group-hover:scale-110 transition-transform duration-300"
                        style={{
                          background: `linear-gradient(135deg, ${step.color}20, ${step.color}10)`
                        }}
                      >
                        <Icon className="w-10 h-10" style={{ color: step.color }} />
                        
                        {/* Glow Effect */}
                        <div 
                          className="absolute inset-0 rounded-xl blur-lg opacity-50"
                          style={{ backgroundColor: step.color }}
                        ></div>
                      </div>

                      {/* Step Number */}
                      <div className="text-center mb-3">
                        <span 
                          className="text-sm font-bold px-3 py-1 rounded-full"
                          style={{
                            background: `${step.color}20`,
                            color: step.color
                          }}
                        >
                          Step {index + 1}
                        </span>
                      </div>

                      <h3 className="text-xl font-bold text-white text-center mb-2">
                        {step.title}
                      </h3>
                      
                      <p className="text-sm text-[#B8C5D6] text-center">
                        {step.description}
                      </p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Mobile Arrows */}
          <div className="md:hidden flex justify-center mt-4">
            <div className="flex flex-col gap-4">
              {[...Array(steps.length - 1)].map((_, i) => (
                <ArrowRight key={i} className="w-6 h-6 text-[#00D9FF] rotate-90 animate-pulse" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
