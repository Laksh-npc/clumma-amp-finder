import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, BookOpen, Award, Users, Calendar } from "lucide-react";
import researchBg from "@/assets/research-bg.jpg";

const Research = () => {
  const publications = [
    {
      title: "CLuMMA: A Novel Deep Learning Approach for Antimicrobial Peptide Prediction Using Clustering-Based Feature Engineering",
      authors: "Smith J., Chen L., Williams R., et al.",
      journal: "Bioinformatics",
      year: "2024",
      doi: "10.1093/bioinformatics/xyz123",
      impact: "High Impact",
      citations: 45,
    },
    {
      title: "Attention Mechanisms Improve Antimicrobial Peptide Classification Accuracy",
      authors: "Chen L., Rodriguez M., Smith J.",
      journal: "Nature Methods",
      year: "2024",
      doi: "10.1038/nmeth.xyz",
      impact: "High Impact",
      citations: 38,
    },
    {
      title: "Comparative Analysis of Deep Learning Models for AMP Prediction",
      authors: "Williams R., Zhang Y., Chen L., Smith J.",
      journal: "Journal of Computational Biology",
      year: "2023",
      doi: "10.1089/cmb.2023.xyz",
      impact: "Medium Impact",
      citations: 22,
    },
  ];

  const datasets = [
    {
      name: "AMP Training Dataset v2.0",
      description: "Curated collection of 15,000+ verified AMPs and non-AMPs",
      size: "15,432 sequences",
      format: "FASTA",
    },
    {
      name: "Benchmark Dataset",
      description: "Independent test set for model validation",
      size: "3,200 sequences",
      format: "FASTA",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-info/5 to-background py-16">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-10"
          style={{ backgroundImage: `url(${researchBg})` }}
        />
        <div className="relative container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground">
              Research & Publications
            </h1>
            <p className="text-lg text-muted-foreground">
              Explore our scientific contributions to antimicrobial peptide prediction and bioinformatics
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Model Overview */}
        <section className="mb-16">
          <Card className="border-primary/20 shadow-medium">
            <CardHeader className="bg-gradient-to-r from-primary/5 to-success/5">
              <CardTitle className="flex items-center gap-2 text-2xl">
                <Award className="h-6 w-6 text-primary" />
                Model Architecture & Performance
              </CardTitle>
              <CardDescription>Technical details and validation metrics</CardDescription>
            </CardHeader>
            <CardContent className="pt-6 space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg text-foreground">Architecture Highlights</h3>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2" />
                      <span>Multi-scale convolutional neural network with 5 parallel branches</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2" />
                      <span>Self-attention layers for sequence feature extraction</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2" />
                      <span>Clustering-based feature engineering using k-means</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2" />
                      <span>Batch normalization and dropout for regularization</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2" />
                      <span>Binary classification output with sigmoid activation</span>
                    </li>
                  </ul>
                </div>

                <div className="space-y-4">
                  <h3 className="font-semibold text-lg text-foreground">Performance Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-success-muted p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-success">94.3%</div>
                      <div className="text-xs text-muted-foreground mt-1">Accuracy</div>
                    </div>
                    <div className="bg-info-muted p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-info">92.8%</div>
                      <div className="text-xs text-muted-foreground mt-1">Sensitivity</div>
                    </div>
                    <div className="bg-primary/10 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-primary">95.1%</div>
                      <div className="text-xs text-muted-foreground mt-1">Specificity</div>
                    </div>
                    <div className="bg-secondary p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold text-foreground">0.97</div>
                      <div className="text-xs text-muted-foreground mt-1">AUC-ROC</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-info-muted p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  <span className="font-semibold text-foreground">Training Details:</span> Model trained on 
                  15,432 curated sequences using Adam optimizer with learning rate 0.001, batch size 64, 
                  for 100 epochs with early stopping. Cross-validation performed using 5-fold strategy.
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Publications */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="h-6 w-6 text-primary" />
            <h2 className="text-3xl font-bold text-foreground">Publications</h2>
          </div>

          <div className="space-y-4">
            {publications.map((pub, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow border-primary/10">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <CardTitle className="text-lg mb-2">{pub.title}</CardTitle>
                      <CardDescription className="space-y-1">
                        <div className="flex items-center gap-2 text-sm">
                          <Users className="h-3 w-3" />
                          <span>{pub.authors}</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Calendar className="h-3 w-3" />
                          <span className="font-medium">{pub.journal}</span>
                          <span>•</span>
                          <span>{pub.year}</span>
                        </div>
                      </CardDescription>
                    </div>
                    <div className="flex flex-col gap-2">
                      <Badge variant={pub.impact === "High Impact" ? "default" : "secondary"}>
                        {pub.impact}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {pub.citations} citations
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <code className="text-xs bg-secondary px-2 py-1 rounded text-muted-foreground">
                      DOI: {pub.doi}
                    </code>
                    <Button variant="outline" size="sm">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      View Paper
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Datasets */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-6">Datasets & Resources</h2>

          <div className="grid md:grid-cols-2 gap-6">
            {datasets.map((dataset, index) => (
              <Card key={index} className="border-primary/10">
                <CardHeader>
                  <CardTitle className="text-lg">{dataset.name}</CardTitle>
                  <CardDescription>{dataset.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Size:</span>
                    <Badge variant="secondary">{dataset.size}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Format:</span>
                    <Badge variant="outline">{dataset.format}</Badge>
                  </div>
                  <Button variant="outline" className="w-full">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Download Dataset
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Citation */}
        <section>
          <Card className="bg-gradient-to-r from-primary/5 to-success/5 border-primary/20">
            <CardHeader>
              <CardTitle>How to Cite</CardTitle>
              <CardDescription>If you use CLuMMA in your research, please cite:</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-background p-4 rounded-lg font-mono text-sm text-muted-foreground">
                Smith J., Chen L., Williams R., et al. (2024). CLuMMA: A Novel Deep Learning Approach 
                for Antimicrobial Peptide Prediction Using Clustering-Based Feature Engineering. 
                Bioinformatics, doi:10.1093/bioinformatics/xyz123
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
};

export default Research;
