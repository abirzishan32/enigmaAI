"use client"

import { Eye, Server, UserX } from "lucide-react"

export function ProblemSection() {
  const problems = [
    {
      icon: Eye,
      title: "Data Exposure",
      description: "LLM agents process sensitive personal information in plaintext, risking unauthorized access and privacy breaches"
    },
    {
      icon: Server,
      title: "Third-Party Risks",
      description: "External AI services collect, store, and potentially misuse user data without transparency"
    },
    {
      icon: UserX,
      title: "Identity Leakage",
      description: "Demographic attributes in prompts enable re-identification attacks and profiling"
    }
  ]

  return (
    <section className="py-20 md:py-32" id="problem">
      <div className="container-custom">
        <div className="text-center mb-16">
          <h2 className="font-bold text-foreground mb-4">
            The Privacy Crisis in AI
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Modern AI systems face critical privacy challenges that put user data at risk
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {problems.map((problem, index) => {
            const Icon = problem.icon
            return (
              <div 
                key={index}
                className="group p-8 rounded-lg border border-border bg-card hover:shadow-premium transition-smooth"
              >
                <div className="mb-6">
                  <div className="inline-flex p-3 rounded-lg bg-primary/10">
                    <Icon className="h-6 w-6 text-primary" strokeWidth={2} />
                  </div>
                </div>
                
                <h3 className="text-xl font-semibold mb-3">
                  {problem.title}
                </h3>
                
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {problem.description}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
