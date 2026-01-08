"use client"

import { Shield, Github, Globe, Box, FileText, Lock } from "lucide-react"

export function FeaturesSection() {
  const features = [
    {
      icon: Shield,
      title: "Post-Quantum Secure",
      description: "CKKS lattice-based encryption resistant to quantum attacks"
    },
    {
      icon: Github,
      title: "Open Source",
      description: "Fully transparent codebase available on GitHub"
    },
    {
      icon: Globe,
      title: "GDPR Compliant",
      description: "Built-in compliance with data protection regulations"
    },
    {
      icon: Box,
      title: "Docker Deployment",
      description: "One-command deployment with full isolation"
    },
    {
      icon: Lock,
      title: "Zero Trust",
      description: "Server never accesses plaintext data"
    }
  ]

  return (
    <section className="py-20 md:py-32 bg-muted/30" id="features">
      <div className="container-custom">
        <div className="text-center mb-16">
          <h2 className="font-bold text-foreground mb-4">
            Enterprise-Grade Features
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built on proven technology with comprehensive privacy protection
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div 
                key={index}
                className="p-6 rounded-lg border border-border bg-card hover:shadow-premium transition-smooth"
              >
                <div className="mb-4">
                  <div className="inline-flex p-2.5 rounded-lg bg-primary/10">
                    <Icon className="h-5 w-5 text-primary" strokeWidth={2} />
                  </div>
                </div>
                
                <h3 className="text-lg font-semibold mb-2">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {feature.description}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
