void DrawDistributions()
{
  gErrorIgnoreLevel = kWarning;
  Int_t runnumber = 242;

  // Open the ROOT file
  auto f = new TFile("data/training-0.root", "READ");
  if (!f || f->IsZombie())
  {
    cout << "Error: Cannot open data/training-0.root" << endl;
    return;
  }

  // Get the tree
  TTree *tree = (TTree *)f->Get("T");
  if (!tree)
  {
    cout << "Error: Cannot find tree 'T' in file" << endl;
    f->Close();
    return;
  }

  // Get list of branches
  TObjArray *branches = tree->GetListOfBranches();

  // Collect branch names matching our patterns
  vector<TString> time_branches, area_branches, fwhm_branches;

  for (Int_t i = 0; i < branches->GetEntries(); i++)
  {
    TString branch_name = branches->At(i)->GetName();
    if (branch_name.BeginsWith("time_"))
    {
      time_branches.push_back(branch_name);
    }
    else if (branch_name.BeginsWith("area_"))
    {
      area_branches.push_back(branch_name);
    }
    else if (branch_name.BeginsWith("fwhm_"))
    {
      fwhm_branches.push_back(branch_name);
    }
  }

  cout << "Found " << time_branches.size() << " time branches" << endl;
  cout << "Found " << area_branches.size() << " area branches" << endl;
  cout << "Found " << fwhm_branches.size() << " fwhm branches" << endl;

  // Create canvas
  auto c = new TCanvas("c", "Distributions", 600, 600);
  c->SetGrid();

  TString pdf_name = Form("plots/Distributions-run%d.pdf", runnumber);
  Int_t page_count = 0;

  // Plot time branches
  for (auto &branch_name : time_branches)
  {
    tree->Draw(Form("%s>>h_%s", branch_name.Data(), branch_name.Data()), "", "");
    TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s", branch_name.Data()));
    if (h)
    {
      h->SetTitle(Form("%s Distribution;%s;Counts", branch_name.Data(), branch_name.Data()));
      h->SetLineColor(kBlue);
      h->SetLineWidth(2);
      h->Draw();
      c->Update();

      if (page_count == 0)
      {
        c->Print(Form("%s(", pdf_name.Data()));
      }
      else
      {
        c->Print(pdf_name.Data());
      }
      page_count++;
    }
  }

  // Plot area branches
  for (auto &branch_name : area_branches)
  {
    tree->Draw(Form("%s>>h_%s", branch_name.Data(), branch_name.Data()), "", "");
    TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s", branch_name.Data()));
    if (h)
    {
      h->SetTitle(Form("%s Distribution;%s;Counts", branch_name.Data(), branch_name.Data()));
      h->SetLineColor(kRed);
      h->SetLineWidth(2);
      h->Draw();
      c->Update();

      if (page_count == 0)
      {
        c->Print(Form("%s(", pdf_name.Data()));
      }
      else
      {
        c->Print(pdf_name.Data());
      }
      page_count++;
    }
  }

  // Plot fwhm branches
  for (size_t i = 0; i < fwhm_branches.size(); i++)
  {
    auto &branch_name = fwhm_branches[i];
    tree->Draw(Form("%s>>h_%s", branch_name.Data(), branch_name.Data()), "", "");
    TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s", branch_name.Data()));
    if (h)
    {
      h->SetTitle(Form("%s Distribution;%s;Counts", branch_name.Data(), branch_name.Data()));
      h->SetLineColor(kGreen + 2);
      h->SetLineWidth(2);
      h->Draw();
      c->Update();

      // Close PDF on the last page
      if (i == fwhm_branches.size() - 1)
      {
        c->Print(Form("%s)", pdf_name.Data()));
      }
      else
      {
        if (page_count == 0)
        {
          c->Print(Form("%s(", pdf_name.Data()));
        }
        else
        {
          c->Print(pdf_name.Data());
        }
      }
      page_count++;
    }
  }

  cout << "Created " << page_count << " pages in " << pdf_name << endl;

  f->Close();
}
