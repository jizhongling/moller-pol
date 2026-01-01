void DrawWaveform()
{
  gErrorIgnoreLevel = kError;
  Int_t runnumber = 242;
  const Int_t method[2] = {2, 0}; // Cluster with method[0], recluster with method[1]
  const Int_t start_type = 1;     // 0: Do not recluster; 1: Recluster with method[1]

  const UInt_t mode = 10;
  const UInt_t slot = 3;
  const UInt_t NUMSAMPLE = 240 / 4;
  const UInt_t ns0 = 40;
  const UInt_t ns1 = 120;
  const UInt_t ng = 50;
  const UInt_t threshold = 600;

  TGraph *g_sample[8];
  for (Int_t ig = 0; ig < 8; ig++)
    g_sample[ig] = new TGraph(ng);

  TH1 *h_spectrum[8];
  for (Int_t ih = 0; ih < 8; ih++)
    h_spectrum[ih] = new TH1F(Form("h_spectrum_%d", ih), Form("ADC spectrum for PMT %d", ih + 1), 90, 10.5, 100.5);

  const char *type[3] = {"Left", "Right", "All"};
  TH1 *h_sum[3];
  for (Int_t ih = 0; ih < 3; ih++)
    h_sum[ih] = new TH1F(Form("h_sum_%d", ih), Form("ADC spectrum for %s PMT Sum", type[ih]), 290, 10.5, 300.5);

  ifstream infile;
  Int_t event, label;
  map<Int_t, Int_t> event_label;
  multimap<Int_t, Int_t> label_event;
  map<Int_t, Int_t> label_count;

  for (Int_t type = start_type; type >= 0; type--)
  {
    TString label_file = Form("data/labels-type%d-method%d.txt", type, method[type]);
    infile.open(label_file.Data());
    if (!infile.is_open())
      continue;
    cout << "Open labels file: " << label_file << endl;
    while (infile >> event >> label)
      if (event_label.find(event) == event_label.end())
      {
        if (type == 1)
          label += 100;
        event_label[event] = label;
        label_event.insert(pair<Int_t, Int_t>(label, event));
        if (label_count.find(label) == label_count.end())
          label_count[label] = 1;
      }
    infile.close();
  }

  auto f = new TFile(Form("Rootfiles/fadc_data_%d.root", runnumber));
  TDirectory *dir = (TDirectory *)f->Get(Form("/mode_%u_data/slot_%u", mode, slot));
  TTree *t_store = (TTree *)dir->Get("waveform");
  UInt_t store_event, store_channel, store_sample[100];
  t_store->SetBranchAddress("event", &store_event);
  t_store->SetBranchAddress("channel", &store_channel);
  t_store->SetBranchAddress("sample", store_sample);

  UInt_t last_event = 0;
  UInt_t total_channel = 0;
  UInt_t fadc_channel = 0;
  vector<UInt_t> v_channel;
  UInt_t max_index[8] = {};
  UInt_t max_sample[8] = {};
  UInt_t max_sample_1 = 0;
  UInt_t max_sample_2 = 0;
  UInt_t event_1or2 = 0;
  UInt_t event_1and2 = 0;
  Float_t sum_sample[8] = {};
  bool trig[2] = {};

  cout << "Plotting waveform for run " << runnumber << ", classified by method " << method[0] << " and " << method[1] << endl;
  TString wavefile = Form("plots/Waveform-run%d-method%d-method%d", runnumber, method[0], method[1]);

  for (ULong64_t ien = 0; ien < t_store->GetEntries(); ien++)
  {
    t_store->GetEntry(ien);

    if (store_event != last_event)
    {
      last_event = store_event;
      ien--;

      if (trig[0] && trig[1])
      {
        for (ULong64_t jen = ien + 1 - total_channel; jen < ien + 1; jen++)
        {
          t_store->GetEntry(jen);
          // cout << store_event << ", " << store_channel << ", " << store_sample[0] << endl;
          if (store_channel < 8)
          {
            v_channel.push_back(store_channel);
            for (UInt_t sample_num = 0; sample_num < ng; sample_num++)
              g_sample[store_channel]->SetPoint((Int_t)sample_num, sample_num * 4, store_sample[sample_num]);
          } // PMT channels
        } // jen

        if (event_label.find(store_event) == event_label.end())
          continue;
        Int_t label = event_label[store_event];

        auto ctmp = new TCanvas(Form("c_label%d_event%u", label, store_event), Form("c_label%d_event%u", label, store_event), 600, 600);
        ctmp->cd();
        auto leg0 = new TLegend(0.1, 0.65, 0.25, 0.9);
        for (UInt_t ich = 0; ich < v_channel.size(); ich++)
        {
          UInt_t chan = v_channel.at(ich);
          g_sample[chan]->SetTitle(Form("Label %d, Size %lu, Event %u", label, label_event.count(label), store_event));
          g_sample[chan]->GetXaxis()->SetTitle("Time (ns)");
          g_sample[chan]->GetYaxis()->SetRangeUser(0, 1600);
          g_sample[chan]->SetLineColor(chan + 1);
          g_sample[chan]->SetLineStyle(1);
          g_sample[chan]->SetLineWidth(3);
          g_sample[chan]->Draw(ich == 0 ? "AL" : "L");
          leg0->AddEntry(g_sample[chan], Form("CH%u", chan), "L");
        }
        leg0->Draw();

        // cout << "Event " << store_event << ", Label " << label << endl;
        if (label_count[label] == 1)
          ctmp->Print(wavefile + Form("-label%d.pdf(", label));
        else
          ctmp->Print(wavefile + Form("-label%d.pdf", label));
        label_count[label]++;
        ctmp->Close();
        delete ctmp;
        delete leg0;
        v_channel.clear();

        for (Int_t ic = 0; ic < 8; ic++)
          h_spectrum[ic]->Fill(sum_sample[ic]);

        UInt_t sum_left = 0;
        for (Int_t ic = 0; ic < 4; ic++)
          sum_left += sum_sample[ic];
        h_sum[0]->Fill(sum_left);

        UInt_t sum_right = 0;
        for (Int_t ic = 4; ic < 8; ic++)
          sum_right += sum_sample[ic];
        h_sum[1]->Fill(sum_right);
        h_sum[2]->Fill(sum_left + sum_right);
      } // trig[0] && trig[1]

      if (max_sample_1 > threshold || max_sample_2 > threshold)
        event_1or2++;
      if (max_sample_1 > threshold && max_sample_2 > threshold)
        event_1and2++;

      total_channel = -1;
      fadc_channel = 0;
      for (Int_t ic = 0; ic < 8; ic++)
      {
        max_index[ic] = 0;
        max_sample[ic] = 0;
        max_sample_1 = 0;
        max_sample_2 = 0;
        sum_sample[ic] = 0;
      }
      for (Int_t it = 0; it < 2; it++)
        trig[it] = 0;
    } // new event

    else if (store_channel < 8)
    {
      for (Int_t is = 0; is < NUMSAMPLE; is++)
      {
        if (store_sample[is] > max_sample[store_channel])
        {
          max_index[store_channel] = is;
          max_sample[store_channel] = store_sample[is];
        }
        if (is >= ns0 / 4 && is < ns1 / 4)
          sum_sample[store_channel] += store_sample[is];
      }
      for (Int_t is = 40 / 4; is < 120 / 4; is++)
        if (store_sample[is] > max_sample_1)
          max_sample_1 = store_sample[is];
      for (Int_t is = 120 / 4; is < 200 / 4; is++)
        if (store_sample[is] > max_sample_2)
          max_sample_2 = store_sample[is];

      const Int_t nped = 4;
      Float_t ped = 0.;
      for (Int_t is = 0; is < nped; is++)
        ped += store_sample[is];
      max_sample[store_channel] -= (Float_t)ped / nped;
      sum_sample[store_channel] /= (Float_t)(ns1 - ns0) / 4;
      sum_sample[store_channel] -= (Float_t)ped / nped;
      sum_sample[store_channel] *= (Float_t)(ns1 - ns0) / 4 / NUMSAMPLE;

      if (max_sample[store_channel] > threshold)
      {
        fadc_channel++;
        trig[store_channel / 4] = true;
      }
    } // PMT channels

    total_channel++;
  } // ien

  // Build Table of Contents based on label_count and render as a ROOT page
  // Compute start page for each label in the final combined PDF (TOC occupies page 1)
  const Int_t toc_pages = 1;
  std::map<Int_t, Int_t> label_start_page;
  Int_t current_page = toc_pages + 1; // first label starts after TOC
  for (auto const &kv : label_count)
  {
    Int_t label = kv.first;
    Int_t cnt = kv.second;
    if (cnt > 1)
    {
      label_start_page[label] = current_page;
      current_page += cnt;
    }
  }

  // Create TOC canvas
  TString tocfile = wavefile + "-TOC.pdf";
  if (!label_start_page.empty())
  {
    auto c_toc = new TCanvas("c_toc", "Table of Contents", 600, 600);
    c_toc->cd();

    auto pave = new TPaveText(0.02, 0.02, 0.98, 0.95, "NDC");
    pave->SetFillColor(0);
    pave->SetBorderSize(1);
    pave->SetTextAlign(12); // left-middle
    pave->SetTextFont(42);

    // Column headers
    auto col1 = new TText(0.07, 0.97, "Label");
    col1->SetTextFont(62);
    col1->SetTextSize(0.028);
    col1->SetNDC();
    col1->Draw();
    auto col2 = new TText(0.20, 0.97, "Start Page");
    col2->SetTextFont(62);
    col2->SetTextSize(0.028);
    col2->SetNDC();
    col2->Draw();

    // Entries
    pave->SetTextSize(0.022);
    for (auto const &kv : label_start_page)
    {
      Int_t label = kv.first;
      Int_t start = kv.second;
      pave->AddText(Form("%6d                 %6d", label, start));
    }

    pave->Draw();
    c_toc->Print(tocfile);
    c_toc->Close();
    delete c_toc;
    delete pave;
    delete col1;
    delete col2;
  }

  TString file_list = tocfile + " ";
  for (const auto &kv : label_count)
    if (kv.second > 1)
    {
      Int_t label = kv.first;
      auto cclose = new TCanvas(Form("c_close_label%d", label), "", 1, 1);
      cclose->Print(wavefile + Form("-label%d.pdf)", label));
      delete cclose;
      file_list += wavefile + Form("-label%d.pdf ", label);
    }
  gSystem->Exec(Form("pdfunite %s %s.pdf", file_list.Data(), wavefile.Data()));
  gSystem->Exec(Form("rm %s", file_list.Data()));

  cout << "OR event = " << event_1or2 << "; AND event = " << event_1and2 << endl;

  // Print instruction for adding clickable TOC links (run outside container)
  if (!label_start_page.empty())
  {
    cout << "\n==================================================" << endl;
    cout << "To add clickable links to TOC, run this command:" << endl;
    cout << "  python add_toc_links.py " << wavefile << ".pdf 0" << endl;
    cout << "==================================================" << endl;
  }

  TString specfile = Form("plots/Spectrum-run%d-%dto%dns.pdf", runnumber, ns0, ns1);
  auto c1 = new TCanvas("c1", "c1", 4 * 600, 2 * 600);
  c1->Divide(4, 2);
  for (Int_t ih = 0; ih < 8; ih++)
  {
    c1->cd(ih + 1);
    h_spectrum[ih]->Draw("HIST");
  }
  c1->Print(specfile + "(");

  auto c2 = new TCanvas("c2", "c2", 2 * 600, 2 * 600);
  c2->Divide(2, 2);
  for (Int_t ih = 0; ih < 3; ih++)
  {
    c2->cd(ih + 1);
    gPad->SetLogy();
    h_sum[ih]->Draw("HIST");
  }
  c2->Print(specfile + ")");
}
