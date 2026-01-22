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
            waveOpacity={0.9}
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

                
            </div>
        </WavyBackground>
    </section>
  )
}

