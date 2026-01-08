"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, Lock, Shuffle, Cpu } from "lucide-react"
import Link from "next/link"
import { Vortex } from "@/components/ui/vortex"

export function HowItWorksSection() {
  return (
    <section className="w-full bg-black overflow-hidden relative">
      <Vortex
         backgroundColor="black"
         rangeY={800}
         particleCount={500}
         baseHue={220}
         className="flex items-center flex-col justify-center px-2 md:px-10 py-10 w-full h-full"
      >
        <div className="container-custom relative z-10">
            <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-24">
            
            <div className="flex-1 space-y-6 text-center lg:text-left">
                <h2 className="text-3xl md:text-5xl font-bold text-white leading-tight">
                Privacy by Design, <br />
                <span className="text-cyan-400">Powered by Mathematics</span>
                </h2>
                <p className="text-lg text-white/80 leading-relaxed max-w-xl">
                Our EnigmaAI platform utilizes <strong>Fully Homomorphic Encryption (FHE)</strong> and advanced shuffling algorithms to ensure your data never leaves its encrypted state during processing.
                </p>
                <p className="text-lg text-white/80 leading-relaxed max-w-xl">
                Whether it's classifying digits or analyzing sensitive records, the server only sees random noise, while you get the precise results you need.
                </p>
                
                <div className="pt-4 flex justify-center lg:justify-start">
                    <Link href="/documentation">
                        <Button size="lg" className="bg-white text-black hover:bg-white/90 gap-2 rounded-full px-8">
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
                        <div className="bg-black/40 backdrop-blur-md border border-white/10 p-6 rounded-2xl shadow-xl hover:border-cyan-500/50 transition-colors">
                            <Lock className="w-8 h-8 text-cyan-400 mb-2" />
                            <h3 className="font-semibold text-white">Encryption</h3>
                            <p className="text-sm text-white/60">Data is locked at the source.</p>
                        </div>
                        <div className="bg-black/40 backdrop-blur-md border border-white/10 p-6 rounded-2xl shadow-xl hover:border-purple-500/50 transition-colors">
                            <Cpu className="w-8 h-8 text-purple-400 mb-2" />
                            <h3 className="font-semibold text-white">Blind Compute</h3>
                            <p className="text-sm text-white/60">Operations on ciphertext.</p>
                        </div>
                    </div>
                    <div className="space-y-4">
                        <div className="bg-black/40 backdrop-blur-md border border-white/10 p-6 rounded-2xl shadow-xl hover:border-emerald-500/50 transition-colors">
                            <Shuffle className="w-8 h-8 text-emerald-400 mb-2" />
                            <h3 className="font-semibold text-white">Shuffling</h3>
                            <p className="text-sm text-white/60">Pattern obfuscation.</p>
                        </div>
                        <div className="bg-gradient-to-br from-indigo-600 to-purple-600 p-6 rounded-2xl shadow-lg text-white flex flex-col justify-center border border-white/10">
                            <span className="text-4xl font-bold mb-1 drop-shadow-md">100%</span>
                            <span className="text-indigo-100 text-sm">Privacy Preserved</span>
                        </div>
                    </div>
                </div>
            </div>

            </div>
        </div>
      </Vortex>
    </section>
  )
}
