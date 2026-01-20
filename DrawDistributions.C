void DrawDistributions()
{
  gErrorIgnoreLevel = kWarning;
  Int_t runnumber = 242;
  const Int_t method[2] = {2, 0}; // Cluster with method[0], recluster with method[1]
  const Int_t start_type = 0;     // 0: Do not recluster; 1: Recluster with method[1]

  // Read labels from file
  ifstream infile;
  Int_t event, label;
  map<Int_t, Int_t> event_label;
  set<Int_t> unique_labels;

  for (Int_t type = start_type; type >= 0; type--)
  {
    TString label_file = Form("data/labels-type%d-method%d.txt", type, method[type]);
    infile.open(label_file.Data());
    if (!infile.is_open())
      continue;
    cout << "Open labels file: " << label_file << endl;
    while (infile >> event >> label)
    {
      if (event_label.find(event) == event_label.end())
      {
        if (type == 1)
          label += 100;
        event_label[event] = label;
        unique_labels.insert(label);
      }
    }
    infile.close();
  }

  cout << "Found " << unique_labels.size() << " unique labels" << endl;

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

  // Set up branch addresses
  Int_t tree_event;
  tree->SetBranchAddress("event", &tree_event);

  // Build branch name lists for Gaussian parameters
  const Int_t chlist[4] = {0, 1, 4, 5};
  vector<TString> amplitude_branches, mean_branches, sigma_branches;
  for (Int_t ich = 0; ich < 4; ich++)
  {
    for (Int_t ip = 0; ip < 2; ip++)
    {
      amplitude_branches.push_back(Form("plot_amplitude_ch%d_p%d", chlist[ich], ip));
      mean_branches.push_back(Form("plot_mean_ch%d_p%d", chlist[ich], ip));
      sigma_branches.push_back(Form("plot_sigma_ch%d_p%d", chlist[ich], ip));
    }
  }

  cout << "Will plot " << amplitude_branches.size() << " amplitude, mean, and sigma distributions per label" << endl;

  TString pdf_name = Form("plots/Distributions-run%d-method%d-method%d.pdf", runnumber, method[0], method[1]);
  auto c = new TCanvas("c", "Distributions", 600, 600);
  c->SetGrid();
  c->SetLogy();

  Int_t page_count = 0;

  // Loop over each label
  for (auto label_id : unique_labels)
  {
    cout << "Processing label " << label_id << endl;

    // Build event list for this label
    vector<Long64_t> event_indices;
    for (Long64_t i = 0; i < tree->GetEntries(); i++)
    {
      tree->GetEntry(i);
      if (event_label.find(tree_event) != event_label.end() && event_label[tree_event] == label_id)
      {
        event_indices.push_back(i);
      }
    }

    if (event_indices.empty())
    {
      cout << "  No events found for label " << label_id << endl;
      continue;
    }

    cout << "  Found " << event_indices.size() << " events for label " << label_id << endl;

    // Create an entry list for this label
    TString entrylist_name = Form("entrylist_label%d", label_id);
    TEntryList *elist = new TEntryList(entrylist_name, entrylist_name);
    for (auto idx : event_indices)
    {
      elist->Enter(idx);
    }
    tree->SetEntryList(elist);

    // Plot amplitude branches for this label
    for (auto &branch_name : amplitude_branches)
    {
      tree->Draw(Form("%s>>h_%s_label%d(100,0,2500)", branch_name.Data(), branch_name.Data(), label_id), "", "");
      TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s_label%d", branch_name.Data(), label_id));
      if (h && h->GetEntries() > 0)
      {
        h->SetTitle(Form("Label %d: %s Distribution;%s;Counts", label_id, branch_name.Data(), branch_name.Data()));
        h->SetLineColor(kBlue);
        h->SetLineWidth(2);
        h->Draw();
        c->Update();

        if (page_count == 0)
          c->Print(Form("%s(", pdf_name.Data()));
        else
          c->Print(pdf_name.Data());
        page_count++;
        delete h;
      }
    }

    // Plot mean branches for this label
    for (auto &branch_name : mean_branches)
    {
      tree->Draw(Form("%s>>h_%s_label%d(100,0,40)", branch_name.Data(), branch_name.Data(), label_id), "", "");
      TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s_label%d", branch_name.Data(), label_id));
      if (h && h->GetEntries() > 0)
      {
        h->SetTitle(Form("Label %d: %s Distribution;%s;Counts", label_id, branch_name.Data(), branch_name.Data()));
        h->SetLineColor(kRed);
        h->SetLineWidth(2);
        h->Draw();
        c->Update();

        c->Print(pdf_name.Data());
        page_count++;
        delete h;
      }
    }

    // Plot sigma branches for this label
    for (auto &branch_name : sigma_branches)
    {
      tree->Draw(Form("%s>>h_%s_label%d(100,0,6)", branch_name.Data(), branch_name.Data(), label_id), "", "");
      TH1F *h = (TH1F *)gDirectory->Get(Form("h_%s_label%d", branch_name.Data(), label_id));
      if (h && h->GetEntries() > 0)
      {
        h->SetTitle(Form("Label %d: %s Distribution;%s;Counts", label_id, branch_name.Data(), branch_name.Data()));
        h->SetLineColor(kGreen + 2);
        h->SetLineWidth(2);
        h->Draw();
        c->Update();

        c->Print(pdf_name.Data());
        page_count++;
        delete h;
      }
    }

    delete elist;
    tree->SetEntryList(nullptr);
  }

  // Close the PDF
  if (page_count > 0)
  {
    c->Print(Form("%s]", pdf_name.Data()));
    cout << "Created " << page_count << " pages in " << pdf_name << endl;
  }
  else
  {
    cout << "No distributions were created" << endl;
  }

  delete c;
  f->Close();
}
