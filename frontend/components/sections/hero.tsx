"use client"

import { Shield, Github, FileText, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col">
      {/* Hero Content */}

      {/* Hero Content */}
      <div className="flex-1 flex items-center">
        <div className="container-custom py-20 md:py-32">
          <div className="max-w-4xl mx-auto text-center space-y-8">
           

            {/* Main Headline */}
            <h1 className="font-bold text-foreground">
              Privacy-Preserving AI
              <br />
              for the Modern Era
            </h1>

            {/* Subheadline */}
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Industry-first framework combining homomorphic encryption and attribute shuffling 
              to protect sensitive data in LLM agent interactions.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Button size="lg" className="h-12 px-8">
                Get Started
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            {/* Trust Indicators */}
            <div className="flex flex-wrap justify-center gap-8 pt-8 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                <span>128-bit Security</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                <span>Open Source</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                <span>GDPR Compliant</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
