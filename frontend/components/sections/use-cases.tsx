"use client"

import { Heart, DollarSign, Scale, User, Shield, Lock, Database, Cloud } from "lucide-react"

export function UseCasesSection() {
  const useCases = [
    {
      icon: Heart,
      title: "Healthcare",
      description: "Encrypt medical records for AI diagnosis without exposing PHI"
    },
    {
      icon: DollarSign,
      title: "Finance",
      description: "Process financial queries without revealing account details"
    },
    {
      icon: Scale,
      title: "Legal",
      description: "Analyze contracts with attorney-client privilege preserved"
    },
    {
      icon: User,
      title: "Personal",
      description: "Generate images without demographic profiling"
    },
    {
      icon: Shield,
      title: "Government",
      description: "Secure citizen data processing with full privacy guarantees"
    },
    {
      icon: Lock,
      title: "Enterprise",
      description: "Corporate AI tools with end-to-end encryption"
    },
    {
      icon: Database,
      title: "Research",
      description: "Analyze sensitive datasets without data exposure"
    },
    {
      icon: Cloud,
      title: "Cloud Services",
      description: "Privacy-first cloud AI with zero-knowledge architecture"
    }
  ]

  // Duplicate for seamless scroll
  const duplicatedUseCases = [...useCases, ...useCases, ...useCases]

  return (
    <section className="py-20 md:py-32" id="use-cases">
      <div className="container-custom">
        <div className="text-center mb-16">
          <h2 className="font-bold text-foreground mb-4">
            Industry Applications
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Privacy protection across all sectors
          </p>
        </div>

        {/* Infinite Scroll Marquee */}
        <div className="relative overflow-hidden">
          <div className="flex gap-4 animate-marquee-slow">
            {duplicatedUseCases.map((useCase, index) => {
              const Icon = useCase.icon
              return (
                <div 
                  key={index} 
                  className="flex-shrink-0 w-[280px] p-6 rounded-lg border border-border bg-card hover:shadow-premium transition-smooth"
                >
                  <div className="mb-4">
                    <div className="inline-flex p-2.5 rounded-lg bg-primary/10">
                      <Icon className="h-5 w-5 text-primary" strokeWidth={2} />
                    </div>
                  </div>

                  <h3 className="text-lg font-semibold mb-2">
                    {useCase.title}
                  </h3>

                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {useCase.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes marquee-slow {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-33.333%);
          }
        }

        .animate-marquee-slow {
          animation: marquee-slow 60s linear infinite;
        }

        .animate-marquee-slow:hover {
          animation-play-state: paused;
        }
      `}</style>
    </section>
  )
}
