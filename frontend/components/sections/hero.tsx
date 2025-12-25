"use client"

import { Shield, Github } from "lucide-react"
import { Button } from "@/components/ui/button"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col bg-gradient-to-b from-[#0F1419] to-[#1A1F2E] overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-20 w-72 h-72 bg-[#00D9FF] rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-[#00FFB3] rounded-full blur-[140px] animate-pulse delay-1000"></div>
      </div>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 backdrop-blur-lg bg-[#0A1628]/80 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Shield className="w-8 h-8 text-[#00D9FF]" />
              <div className="absolute inset-0 bg-[#00D9FF] blur-md opacity-50"></div>
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] bg-clip-text text-transparent">
              EnigmaAI
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#home" className="text-[#B8C5D6] hover:text-white transition-colors">Home</a>
            <a href="#features" className="text-[#B8C5D6] hover:text-white transition-colors">Features</a>
            <a href="#technology" className="text-[#B8C5D6] hover:text-white transition-colors">Technology</a>
            <a href="#docs" className="text-[#B8C5D6] hover:text-white transition-colors">Docs</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-[#B8C5D6] hover:text-white transition-colors">
              <Github className="w-5 h-5" />
            </a>
          </div>

          <Button 
            className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-black font-semibold shadow-lg shadow-[#00D9FF]/50 transition-all hover:shadow-xl hover:shadow-[#00D9FF]/60"
          >
            Try Demo
          </Button>
        </div>
      </nav>

      {/* Hero Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-20 relative z-10">
        <div className="max-w-5xl mx-auto text-center space-y-8">
          {/* Main Headline */}
          <h1 className="text-5xl md:text-7xl font-bold leading-tight">
            <span className="bg-gradient-to-r from-[#00D9FF] via-[#00FFB3] to-[#00D9FF] bg-clip-text text-transparent animate-gradient bg-[length:200%_auto]">
              Safeguarding Your Privacy
            </span>
            <br />
            <span className="text-white">in the Age of AI</span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-[#B8C5D6] max-w-3xl mx-auto leading-relaxed">
            Industry-first privacy-preserving framework for tool-using LLM agents 
            using homomorphic encryption and t-closeness attribute shuffling
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
            <Button 
              size="lg"
              className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-black font-semibold text-lg px-8 py-6 shadow-lg shadow-[#00D9FF]/50 hover:shadow-xl hover:shadow-[#00D9FF]/60 transition-all"
            >
              Get Started
            </Button>
            <Button 
              size="lg"
              variant="outline"
              className="border-2 border-white/30 text-white hover:bg-white/10 text-lg px-8 py-6 backdrop-blur-sm transition-all"
            >
              Read Research Paper
            </Button>
          </div>

          {/* Trust Badges */}
          <div className="flex flex-wrap justify-center items-center gap-6 pt-8">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="w-2 h-2 bg-[#FFD700] rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-white">IEEE TDSC 2024</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="w-2 h-2 bg-[#00FFB3] rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-white">128-bit Security</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-sm border border-white/10">
              <div className="w-2 h-2 bg-[#00D9FF] rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-white">Open Source</span>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Code Snippets Effect */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-1/4 left-10 text-[#00D9FF] font-mono text-sm animate-float">
          encrypt(data)
        </div>
        <div className="absolute top-1/3 right-20 text-[#00FFB3] font-mono text-sm animate-float delay-500">
          {"<lock>"}
        </div>
        <div className="absolute bottom-1/3 left-1/4 text-[#00D9FF] font-mono text-sm animate-float delay-1000">
          t-closeness ≤ 0.3
        </div>
      </div>
    </section>
  )
}
