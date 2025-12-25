"use client"

import { Shield, Github, Globe, Container, FileText, Server } from "lucide-react"

export function FeaturesSection() {
  const features = [
    {
      icon: Shield,
      title: "Post-Quantum Secure",
      description: "CKKS lattice-based encryption resistant to quantum attacks",
      gradient: "from-[#00D9FF] to-[#00FFB3]"
    },
    {
      icon: Github,
      title: "Open Source",
      description: "Fully transparent codebase available on GitHub",
      gradient: "from-[#00FFB3] to-[#FFD700]"
    },
    {
      icon: Globe,
      title: "GDPR Compliant",
      description: "Built-in compliance with data protection regulations",
      gradient: "from-[#FFD700] to-[#00D9FF]"
    },
    {
      icon: Container,
      title: "Docker Deployment",
      description: "One-command deployment with full isolation",
      gradient: "from-[#00D9FF] to-[#00FFB3]"
    },
    {
      icon: FileText,
      title: "Research-Backed",
      description: "Published in IEEE TDSC 2024, peer-reviewed methodology",
      gradient: "from-[#00FFB3] to-[#FFD700]"
    },
    {
      icon: Server,
      title: "Zero Trust",
      description: "Server never accesses plaintext data",
      gradient: "from-[#FFD700] to-[#00D9FF]"
    }
  ]

  return (
    <section className="relative py-24 bg-gradient-to-br from-[#0F1419] to-[#1A1F2E] overflow-hidden">
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Key Features
          </h2>
          <p className="text-xl text-[#B8C5D6]">
            Enterprise-grade privacy protection built on proven technology
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div 
                key={index}
                className="group relative p-6 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:border-[#00D9FF]/50 transition-all duration-300 hover:shadow-xl hover:shadow-[#00D9FF]/20 hover:-translate-y-1"
              >
                {/* Gradient Background on Hover */}
                <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}></div>
                
                <div className="relative z-10">
                  {/* Icon */}
                  <div className={`w-14 h-14 mb-4 rounded-xl bg-gradient-to-br ${feature.gradient} p-0.5 group-hover:scale-110 transition-transform duration-300`}>
                    <div className="w-full h-full bg-[#0A1628] rounded-xl flex items-center justify-center">
                      <Icon className="w-7 h-7 text-[#00D9FF]" />
                    </div>
                  </div>
                  
                  <h3 className="text-xl font-bold text-white mb-2">
                    {feature.title}
                  </h3>
                  
                  <p className="text-[#B8C5D6] text-sm leading-relaxed">
                    {feature.description}
                  </p>
                </div>

                {/* Corner Accent */}
                <div className="absolute top-2 right-2 w-20 h-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className={`absolute top-0 right-0 w-3 h-3 rounded-full bg-gradient-to-br ${feature.gradient}`}></div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
