"use client"

import { Heart, DollarSign, Scale, User, Shield, Lock, Database, Cloud } from "lucide-react"

export function UseCasesSection() {
  const useCases = [
    {
      icon: Heart,
      title: "Healthcare",
      description: "Encrypt medical records for AI diagnosis without exposing PHI",
      color: "#00FFB3"
    },
    {
      icon: DollarSign,
      title: "Finance",
      description: "Process financial queries without revealing account details",
      color: "#FFD700"
    },
    {
      icon: Scale,
      title: "Legal",
      description: "Analyze contracts with attorney-client privilege preserved",
      color: "#00D9FF"
    },
    {
      icon: User,
      title: "Personal",
      description: "Generate images without demographic profiling",
      color: "#00FFB3"
    },
    {
      icon: Shield,
      title: "Government",
      description: "Secure citizen data processing with full privacy guarantees",
      color: "#FFD700"
    },
    {
      icon: Lock,
      title: "Enterprise",
      description: "Corporate AI tools with end-to-end encryption",
      color: "#00D9FF"
    },
    {
      icon: Database,
      title: "Research",
      description: "Analyze sensitive datasets without data exposure",
      color: "#00FFB3"
    },
    {
      icon: Cloud,
      title: "Cloud Services",
      description: "Privacy-first cloud AI with zero-knowledge architecture",
      color: "#FFD700"
    }
  ]

  // Duplicate the array for seamless infinite scroll
  const duplicatedUseCases = [...useCases, ...useCases, ...useCases]

  return (
    <section className="relative py-24 bg-gradient-to-br from-[#1A1F2E] to-[#0F1419] overflow-hidden">
      <div className="relative z-10 w-full px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Real-World Use Cases
          </h2>
          <p className="text-xl text-[#B8C5D6]">
            Privacy protection across industries
          </p>
        </div>

        {/* Infinite Scroll Marquee */}
        <div className="relative w-full overflow-hidden">
          {/* Gradient Overlays for fade effect */}
          <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-[#1A1F2E] to-transparent z-10 pointer-events-none"></div>
          <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-[#0F1419] to-transparent z-10 pointer-events-none"></div>

          {/* Scrolling Container */}
          <div className="group">
            <div className="flex gap-6 animate-marquee group-hover:pause-animation">
              {duplicatedUseCases.map((useCase, index) => {
                const Icon = useCase.icon
                return (
                  <div 
                    key={index} 
                    className="flex-shrink-0 w-[320px] h-[220px] p-6 rounded-2xl bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border border-white/10 hover:border-[#00D9FF]/50 transition-all duration-300 hover:shadow-xl hover:shadow-[#00D9FF]/20 hover:scale-105"
                  >
                    {/* Icon */}
                    <div 
                      className="w-14 h-14 mb-4 rounded-xl flex items-center justify-center"
                      style={{ background: `${useCase.color}20` }}
                    >
                      <Icon className="w-7 h-7" style={{ color: useCase.color }} />
                    </div>

                    {/* Content */}
                    <h3 className="text-xl font-bold text-white mb-3">
                      {useCase.title}
                    </h3>

                    <p className="text-sm text-[#B8C5D6] leading-relaxed">
                      {useCase.description}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes marquee {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-33.333%);
          }
        }

        .animate-marquee {
          animation: marquee 40s linear infinite;
        }

        .pause-animation {
          animation-play-state: paused;
        }
      `}</style>
    </section>
  )
}
