"use client"

import { Shield, Github, FileText, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { WavyBackground } from "@/components/ui/wavy-background"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

export function HeroSection() {
  const { resolvedTheme } = useTheme();
  const [backgroundFill, setBackgroundFill] = useState("black");

  useEffect(() => {
    setBackgroundFill(resolvedTheme === "dark" ? "black" : "white");
  }, [resolvedTheme]);

  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden">
        <WavyBackground 
            className="max-w-4xl mx-auto pb-40"
            containerClassName="min-h-screen"
            colors={["#38bdf8", "#818cf8", "#c084fc", "#e879f9", "#22d3ee"]}
            waveWidth={50}
            backgroundFill={backgroundFill}
            blur={10}
            speed="fast"
            waveOpacity={0.5}
        >
            <div className="text-center space-y-8 px-4">
                {/* Main Headline */}
                <h1 className="text-4xl md:text-7xl font-bold text-black dark:text-white inter-var">
                Privacy-Preserving AI
                <br />
                for the Modern Era
                </h1>

                {/* Subheadline */}
                <p className="text-base md:text-xl text-black/80 dark:text-white/80 max-w-2xl mx-auto font-normal inter-var">
                Industry-first framework combining homomorphic encryption and attribute shuffling 
                to protect sensitive data in LLM agent interactions.
                </p>

                {/* CTA Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
                <Button size="lg" className="h-12 px-8 bg-black dark:bg-white text-white dark:text-black hover:bg-black/90 dark:hover:bg-white/90">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
                </div>

                {/* Trust Indicators */}
                <div className="flex flex-wrap justify-center gap-8 pt-12 text-sm text-black/60 dark:text-white/60">
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                    <span>128-bit Security</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-purple-400 shadow-[0_0_8px_rgba(192,132,252,0.8)]" />
                    <span>Open Source</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-pink-400 shadow-[0_0_8px_rgba(232,121,249,0.8)]" />
                    <span>GDPR Compliant</span>
                </div>
                </div>
            </div>
        </WavyBackground>
    </section>
  )
}

