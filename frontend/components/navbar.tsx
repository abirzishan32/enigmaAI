"use client"

import { Shield, Github } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"
import Link from "next/link"

export function Navbar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="container-custom">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" strokeWidth={2} />
            <span className="text-lg font-semibold tracking-tight">EnigmaAI </span>
          </Link>
          
          <div className="hidden md:flex items-center gap-8">
            <Link href="/#features" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              Features
            </Link>
            <Link href="/#technology" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              Technology
            </Link>
            <Link href="/#research" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              Research
            </Link>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github className="h-5 w-5 text-muted-foreground hover:text-foreground transition-colors" />
            </a>
          </div>

          <div className="flex items-center gap-3">
            <ModeToggle />
            <Button size="sm" className="h-9" asChild>
                <Link href="/digit-recog">Live Demo</Link>
            </Button>
          </div>
        </div>
      </div>
    </nav>
  )
}
