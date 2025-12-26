import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Zap, Shield, BarChart3, FileText } from "lucide-react";
import heroPeptide from "@/assets/hero-peptide.jpg";
import aminoAcids from "@/assets/amino-acids.jpg";

const Home = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-30"
          style={{ backgroundImage: `url(${heroPeptide})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-primary/20 to-background" />
        
        <div className="relative container mx-auto px-4 py-24 md:py-32">
          <div className="max-w-3xl mx-auto text-center space-y-6 animate-fade-in">
            <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight">
              Advanced Antimicrobial Peptide Prediction
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground">
              Harness the power of deep learning to identify AMPs with unprecedented accuracy
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link to="/predict">
                <Button size="lg" className="text-lg px-8 group">
                  Start Prediction
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link to="/research">
                <Button size="lg" variant="outline" className="text-lg px-8">
                  Read Research
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-secondary/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Why Choose CLuMMA?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              State-of-the-art deep learning technology for antimicrobial peptide prediction
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <Card className="hover:shadow-lg transition-shadow animate-fade-in">
              <CardHeader>
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-3">
                  <Zap className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Fast & Accurate</CardTitle>
                <CardDescription>
                  Deep neural networks with attention mechanisms provide rapid, precise predictions
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow animate-fade-in" style={{ animationDelay: "0.1s" }}>
              <CardHeader>
                <div className="w-12 h-12 bg-success/10 rounded-lg flex items-center justify-center mb-3">
                  <Shield className="h-6 w-6 text-success" />
                </div>
                <CardTitle>Validated Model</CardTitle>
                <CardDescription>
                  Trained on curated datasets of verified AMPs and non-AMPs for reliable results
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow animate-fade-in" style={{ animationDelay: "0.2s" }}>
              <CardHeader>
                <div className="w-12 h-12 bg-info/10 rounded-lg flex items-center justify-center mb-3">
                  <BarChart3 className="h-6 w-6 text-info" />
                </div>
                <CardTitle>Confidence Metrics</CardTitle>
                <CardDescription>
                  Get detailed probability scores and confidence levels for each prediction
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* About Model Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center max-w-6xl mx-auto">
            <div className="space-y-6">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Clustering-Based Machine Learning
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                CLuMMA employs advanced deep learning architecture combining multi-scale convolutional 
                layers with attention mechanisms to analyze amino acid sequences for antimicrobial activity.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2" />
                  <div>
                    <h3 className="font-semibold text-foreground">Multi-Scale CNN Architecture</h3>
                    <p className="text-muted-foreground text-sm">
                      Captures features at different sequence scales for comprehensive analysis
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2" />
                  <div>
                    <h3 className="font-semibold text-foreground">Attention Mechanisms</h3>
                    <p className="text-muted-foreground text-sm">
                      Focuses on the most important regions of peptide sequences
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2" />
                  <div>
                    <h3 className="font-semibold text-foreground">Clustering-Based Training</h3>
                    <p className="text-muted-foreground text-sm">
                      Leverages sequence similarity patterns for improved generalization
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="rounded-lg overflow-hidden shadow-xl">
                <img
                  src={aminoAcids}
                  alt="Amino acid sequence visualization"
                  className="w-full h-auto"
                />
              </div>
              <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-gradient-to-br from-primary/20 to-success/20 rounded-lg -z-10" />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary/10 via-primary/5 to-success/10">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Ready to Predict AMPs?
            </h2>
            <p className="text-lg text-muted-foreground">
              Start analyzing your amino acid sequences now with our advanced prediction tool
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link to="/predict">
                <Button size="lg" className="text-lg px-8">
                  Try Prediction Tool
                </Button>
              </Link>
              <Link to="/research">
                <Button size="lg" variant="outline" className="text-lg px-8">
                  <FileText className="mr-2 h-5 w-5" />
                  View Publications
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
