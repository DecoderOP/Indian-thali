export interface UserProfileData {
  disease: string;
  symptoms: string[];
  age: number;
  gender: string;
}

export async function generatePlan(userProfile: UserProfileData) {
  const res = await fetch("http://localhost:8000/api/generate-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userProfile),
    cache: "no-store",
  });

  if (!res.ok) throw new Error("Failed to generate diet plan");

  return res.json();
}
