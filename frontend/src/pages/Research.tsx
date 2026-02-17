import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, BookOpen } from "lucide-react";

const Research = () => {
  const teamMembers = [
    {
      name: "Sainithin Artham",
      affiliation: "BML Munjal University",
    },
    {
      name: "Akash Saraswat",
      affiliation: "BML Munjal University",
    },
    {
      name: "Yugen Jarwal",
      affiliation: "BML Munjal University",
    },
    {
      name: "Laksh Sharda",
      affiliation: "BML Munjal University",
    },
    {
      name: "Bipin Singh",
      affiliation: "Mahindra University, Hyderabad",
    },
    {
      name: "Arijit Maitra",
      affiliation: "BML Munjal University",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Research Team */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <Users className="h-6 w-6 text-primary" />
            <h2 className="text-3xl font-bold text-foreground">Research Team</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {teamMembers.map((member, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow border-primary/10">
                <CardHeader>
                  <CardTitle className="text-lg">{member.name}</CardTitle>
                  <CardDescription className="space-y-1">
                    <div className="text-xs">{member.affiliation}</div>
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>

        {/* Citation */}
        <section>
          <Card className="bg-gradient-to-r from-primary/5 to-success/5 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                How to Cite
              </CardTitle>
              <CardDescription>If you use Pred-Anti Hepatitis C Peptide Prediction-DL in your research, please cite:</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-background p-4 rounded-lg font-mono text-sm text-muted-foreground">
                [Update this to the title of your paper]
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
};

export default Research;
