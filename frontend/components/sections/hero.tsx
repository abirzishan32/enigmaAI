"use client"

import { Shield, Github, FileText, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="container-custom">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" strokeWidth={2} />
              <span className="text-lg font-semibold tracking-tight">EnigmaAI </span>
            </div>
            
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                Features
              </a>
              <a href="#technology" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                Technology
              </a>
              <a href="#research" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                Research
              </a>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                <Github className="h-5 w-5 text-muted-foreground hover:text-foreground transition-colors" />
              </a>
            </div>

            <div className="flex items-center gap-3">
              <ModeToggle />
              <Button size="sm" className="h-9">
                Get Started
              </Button>
            </div>
          </div>
        </div>
      </nav>

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
              <Button size="lg" variant="outline" className="h-12 px-8">
                Read Research
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
