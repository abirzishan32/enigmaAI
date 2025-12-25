"use client"

export function TechnicalSpecsSection() {
  const techStack = [
    "Python 3.10 + PyTorch",
    "TenSEAL (Microsoft SEAL wrapper)",
    "Transformers (Hugging Face)",
    "Docker + CUDA support",
    "FastAPI Backend",
    "React + Next.js Frontend"
  ]

  const metrics = [
    { label: "Security", value: 128, max: 128, color: "#00FFB3", unit: "-bit" },
    { label: "Encrypted Accuracy", value: 85, max: 100, color: "#00D9FF", unit: "%" },
    { label: "Plaintext Accuracy", value: 92, max: 100, color: "#FFD700", unit: "%" },
    { label: "t-Closeness", value: 28, max: 30, color: "#00FFB3", unit: "/30", inverted: true }
  ]

  return (
    <section className="relative py-24 bg-[#0A1628] overflow-hidden">
      {/* Code Pattern Background */}
      <div className="absolute inset-0 opacity-5 font-mono text-xs">
        <div className="absolute top-10 left-10">
          {"const encrypted = seal.encrypt(data);"}
        </div>
        <div className="absolute top-32 right-20">
          {"function shuffle(attrs) { return randomize(attrs); }"}
        </div>
        <div className="absolute bottom-20 left-1/4">
          {"if (tCloseness <= 0.3) { approve(); }"}
        </div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Built on Cutting-Edge Technology
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] mx-auto"></div>
        </div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* LEFT: Technology Stack */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <div className="w-1 h-8 bg-gradient-to-b from-[#00D9FF] to-[#00FFB3]"></div>
              Technology Stack
            </h3>

            {/* Terminal Window */}
            <div className="rounded-xl overflow-hidden border border-white/10 bg-black/50 backdrop-blur-sm">
              {/* Terminal Header */}
              <div className="flex items-center gap-2 px-4 py-3 bg-white/5 border-b border-white/10">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="ml-2 text-xs text-[#B8C5D6]">privacy-asst-stack.sh</span>
              </div>

              {/* Terminal Content */}
              <div className="p-6 font-mono text-sm space-y-3">
                <div className="text-[#00FFB3]">$ cat requirements.txt</div>
                {techStack.map((tech, index) => (
                  <div key={index} className="text-[#B8C5D6] pl-4 animate-fade-in" style={{ animationDelay: `${index * 100}ms` }}>
                    <span className="text-[#00D9FF]">✓</span> {tech}
                  </div>
                ))}
                <div className="text-[#00FFB3] animate-pulse mt-4">▊</div>
              </div>
            </div>
          </div>

          {/* RIGHT: Performance Metrics */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <div className="w-1 h-8 bg-gradient-to-b from-[#FFD700] to-[#00D9FF]"></div>
              Performance Metrics
            </h3>

            <div className="space-y-6">
              {metrics.map((metric, index) => {
                const percentage = metric.inverted 
                  ? ((metric.max - metric.value) / metric.max) * 100 
                  : (metric.value / metric.max) * 100
                
                return (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-white font-semibold">{metric.label}</span>
                      <span className="text-[#B8C5D6] font-mono">
                        {metric.value}{metric.unit}
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="relative h-3 bg-white/5 rounded-full overflow-hidden border border-white/10">
                      <div 
                        className="absolute left-0 top-0 h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${percentage}%`,
                          background: `linear-gradient(90deg, ${metric.color}, ${metric.color}80)`,
                          boxShadow: `0 0 10px ${metric.color}50`
                        }}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
                      </div>
                    </div>

                    {/* Additional Info */}
                    {metric.label === "Encrypted Accuracy" && (
                      <p className="text-xs text-[#B8C5D6] italic">
                        26.7× slower but acceptable for privacy-critical tasks
                      </p>
                    )}
                  </div>
                )
              })}

              {/* Summary Card */}
              <div className="mt-8 p-4 rounded-xl bg-gradient-to-br from-[#00FFB3]/10 to-[#00D9FF]/10 border border-[#00FFB3]/30">
                <p className="text-sm text-white">
                  <span className="font-bold text-[#00FFB3]">Performance Summary:</span> Achieves strong privacy guarantees with acceptable computational overhead for sensitive data applications.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
