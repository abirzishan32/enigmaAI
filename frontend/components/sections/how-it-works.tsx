"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, Lock, Shuffle, Cpu } from "lucide-react"
import Link from "next/link"

export function HowItWorksSection() {
  return (
    <section className="py-24 bg-muted/30 border-y border-border/50">
      <div className="container-custom">
        <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-24">
          
          <div className="flex-1 space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Privacy by Design, <br />
              <span className="text-indigo-600 dark:text-indigo-400">Powered by Mathematics</span>
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Our EnigmaAI platform utilizes <strong>Fully Homomorphic Encryption (FHE)</strong> and advanced shuffling algorithms to ensure your data never leaves its encrypted state during processing.
            </p>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Whether it's classifying digits or analyzing sensitive records, the server only sees random noise, while you get the precise results you need.
            </p>
            
            <div className="pt-4">
                <Link href="/documentation">
                    <Button size="lg" className="bg-foreground text-background hover:bg-foreground/90 gap-2">
                        Learn More About The Technology
                        <ArrowRight className="w-4 h-4" />
                    </Button>
                </Link>
            </div>
          </div>

          <div className="flex-1 w-full relative">
             {/* Abstract Visual Representation */}
             <div className="grid grid-cols-2 gap-4">
                <div className="space-y-4 pt-8">
                    <div className="bg-card border p-6 rounded-2xl shadow-sm">
                        <Lock className="w-8 h-8 text-indigo-500 mb-2" />
                        <h3 className="font-semibold">Encryption</h3>
                        <p className="text-sm text-muted-foreground">Data is locked at the source.</p>
                    </div>
                    <div className="bg-card border p-6 rounded-2xl shadow-sm">
                        <Cpu className="w-8 h-8 text-purple-500 mb-2" />
                        <h3 className="font-semibold">Blind Compute</h3>
                        <p className="text-sm text-muted-foreground">Operations on ciphertext.</p>
                    </div>
                </div>
                <div className="space-y-4">
                    <div className="bg-card border p-6 rounded-2xl shadow-sm">
                        <Shuffle className="w-8 h-8 text-green-500 mb-2" />
                        <h3 className="font-semibold">Shuffling</h3>
                        <p className="text-sm text-muted-foreground">Pattern obfuscation.</p>
                    </div>
                     <div className="bg-indigo-600 p-6 rounded-2xl shadow-lg text-white flex flex-col justify-center">
                        <span className="text-4xl font-bold mb-1">100%</span>
                        <span className="text-indigo-100 text-sm">Privacy Preserved</span>
                    </div>
                </div>
             </div>
          </div>

        </div>
      </div>
    </section>
  )
}
